from envs.hanabi_env import HanabiMadrona, PantheonHanabi, validate_step, config_choice

from pantheonrl_extension.asyncvectorenv import AsyncVectorEnv
from pantheonrl_extension.vectorobservation import VectorObservation

import torch
import time

import argparse

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--verbose", default=False, nargs="?", const=True,
                        help="if toggled, enable assertions to validate correctness")
    parser.add_argument("--asserts", default=False, nargs="?", const=True,
                        help="if toggled, enable assertions to validate correctness")

    parser.add_argument("--use-cpu", default=False, nargs="?", const=True,
                        help="if toggled, use cpu version of madrona")
    parser.add_argument("--use-env-cpu", default=False, nargs="?", const=True,
                        help="if toggled, use cpu for env outputs")
    parser.add_argument("--use-baseline", default=False, nargs="?", const=True,
                        help="if toggled, use baseline version")

    parser.add_argument("--validation", default=False, nargs="?", const=True,
                        help="if toggled, validate correctness")
    parser.add_argument("--debug-compile", default=False, nargs="?", const=True,
                        help="if toggled, use debug compilation mode")
    parser.add_argument("--hanabi-type", type=str, default="full",
                        choices=['very_small', 'small', 'full'],
                        help="Choice for hanabi type.")
    args = parser.parse_args()

    han_conf = config_choice[args.hanabi_type]

    if args.use_baseline:
        env = AsyncVectorEnv(
                [lambda: PantheonHanabi(han_conf) for _ in range(args.num_envs)],
                device=torch.device('cpu') if args.use_env_cpu else None
            )
    else:
        env = HanabiMadrona(args.num_envs, 0, args.debug_compile, han_conf, args.use_cpu, args.use_env_cpu)
    old_state = env.n_reset()
    actions = torch.zeros((2, args.num_envs, 1), dtype=int).to(device=env.device)
    num_errors = 0

    # warp up
    for _ in range(5):
        for i in range(2):
            logits = torch.rand(args.num_envs, env.action_space.n).to(device=env.device)
            logits[torch.logical_not(old_state[i].action_mask)] = -float('inf')
            actions[i, :, 0] = torch.max(logits, dim=1).indices  # torch.randint_like(actions, high=4)
        next_state, reward, next_done, _ = env.n_step(actions)
        old_state = [VectorObservation(s.active.clone(), s.obs.clone(), s.state.clone(), s.action_mask.clone()) for s in next_state]

    time_stamps = [0 for i in range(args.num_steps * 2)]
    for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
        for i in range(2):
            logits = torch.rand(args.num_envs, env.action_space.n).to(device=env.device)
            logits[torch.logical_not(old_state[i].action_mask)] = -float('inf')
            actions[i, :, 0] = torch.max(logits, dim=1).indices  # torch.randint_like(actions, high=4)
        # print(actions)

        time_stamps[iter * 2] = time.time()
        next_state, reward, next_done, _ = env.n_step(actions)
        time_stamps[iter * 2 + 1] = time.time()

        if args.validation and not validate_step(old_state, actions, next_done, next_state, reward, han_conf, verbose=args.verbose):
            num_errors += 1
            assert(not args.asserts)

        old_state = [VectorObservation(s.active.clone(), s.obs.clone(), s.state.clone(), s.action_mask.clone()) for s in next_state]

    time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
    assert(len(time_difference) == args.num_steps)
    print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
    if args.validation:
        print("Error rate:", num_errors/args.num_steps)
    env.close()
