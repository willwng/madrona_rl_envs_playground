# from envs.hanabi_env import HanabiMadrona, PantheonHanabi, validate_step, config_choice
from envs.overcooked_env import PantheonOvercooked, validate_step, init_validation, SimplifiedOvercooked, base_layout_params #, OvercookedMadrona

from pantheonrl_extension.vectorenv import SyncVectorEnv

import torch
import time

import argparse

from tqdm import tqdm

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

parser.add_argument("--use-baseline", default=False, nargs="?", const=True,
        help="if toggled, use baseline version")

parser.add_argument("--validation", default=False, nargs="?", const=True,
        help="if toggled, validate correctness")
parser.add_argument("--debug-compile", default=False, nargs="?", const=True,
        help="if toggled, use debug compilation mode")
parser.add_argument("--layout", type=str, default="cramped_room",
                        choices=['cramped_room', 'coordination_ring', 'asymmetric_advantages_tomato', 'bonus_order_test'],
                        help="Choice for overcooked layout.")
args = parser.parse_args()

print(base_layout_params(args.layout, 400))

if args.use_baseline:
    env = SyncVectorEnv(
            [lambda: SimplifiedOvercooked(args.layout) for _ in range(args.num_envs)]
        )
else:
    # env = OvercookedMadrona(args.layout, args.num_envs, 0, args.debug_compile, args.use_cpu)
    pass

old_state = env.n_reset()
actions = torch.zeros((2, args.num_envs, 1), dtype=int).to(device=env.device)
num_errors = 0

init_validation(args.layout, args.num_envs)

# warp up
for _ in range(5):
    actions[:, :, :] = torch.randint_like(actions, high=env.action_space.n)
    next_state, reward, next_done, _ = env.n_step(actions)

    if args.validation and not validate_step(old_state, actions, next_done, next_state, reward, verbose=args.verbose):
        num_errors += 1
        assert(not args.asserts)

    old_state = next_state

time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    actions[:, :, :] = torch.randint_like(actions, high=env.action_space.n)
    # print(actions)

    time_stamps[iter * 2] = time.time()
    next_state, reward, next_done, _ = env.n_step(actions)
    time_stamps[iter * 2 + 1] = time.time()

    if args.validation and not validate_step(old_state, actions, next_done, next_state, reward, verbose=args.verbose):
        num_errors += 1
        assert(not args.asserts)

    old_state = next_state
    
time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
if args.validation:
    print("Error rate:", num_errors/args.num_steps)
