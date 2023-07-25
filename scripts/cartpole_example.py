from envs.cartpole_env import CartpoleMadronaTorch, validate_step, CartpoleNumpy
import torch
import time

import gym

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
                        help="if toggled, validate correctness")
    args = parser.parse_args()

    if args.use_baseline:
        env = gym.vector.AsyncVectorEnv(
                [lambda: CartpoleNumpy() for _ in range(args.num_envs)],
            )
        actions = torch.zeros((args.num_envs), dtype=int)
    else:
        env = CartpoleMadronaTorch(args.num_envs, 0, args.debug_compile, args.use_cpu, args.use_env_cpu)
        actions = torch.zeros((args.num_envs), dtype=int).to(device=env.device)

    old_state = env.reset()
    num_errors = 0
    start_time = time.time()

    if args.use_baseline:
        old_state = torch.tensor(old_state)
    else:
        old_state = old_state.clone()

    for _ in range(5):
        action = torch.randint_like(actions, high=2)

        next_state, reward, next_done, _ = env.step(action if not args.use_baseline else action[:, 0].cpu().numpy())

        if args.use_baseline:
            next_done = torch.tensor(next_done)
            next_state = torch.tensor(next_state)

        if args.validation and not validate_step(old_state, action, next_done, next_state, verbose=args.verbose):
            num_errors += 1
            assert(not args.asserts)

        old_state = next_state.clone()

    time_stamps = [0 for i in range(args.num_steps * 2)]
    for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
        action = torch.randint_like(actions, high=2)

        time_stamps[iter * 2] = time.time()
        next_state, reward, next_done, _ = env.step(action if not args.use_baseline else action[:, 0].cpu().numpy())
        time_stamps[iter * 2 + 1] = time.time()

        if args.use_baseline:
            next_done = torch.tensor(next_done)
            next_state = torch.tensor(next_state)

        if args.validation and not validate_step(old_state, action, next_done, next_state, verbose=args.verbose):
            num_errors += 1
            assert(not args.asserts)

        old_state = next_state.clone()
    time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
    assert(len(time_difference) == args.num_steps)
    print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
    if args.validation:
        print("Error rate:", num_errors/args.num_steps)

    env.close()
