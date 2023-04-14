from envs.overcooked_env import PantheonOvercooked, validate_step, init_validation, SimplifiedOvercooked, base_layout_params, OvercookedMadrona

from pantheonrl_extension.vectorenv import SyncVectorEnv

import torch
import time

import argparse

from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32,
                    help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=1000,
                    help="the number of steps to run in each environment per policy rollout")

parser.add_argument("--use-cpu", default=False, nargs="?", const=True,
                    help="if toggled, use cpu version of madrona")
parser.add_argument("--use-env-cpu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, use cpu for env outputs")

parser.add_argument("--debug-compile", default=False, nargs="?", const=True,
                    help="if toggled, use debug compilation mode")
parser.add_argument("--layout", type=str, default="cramped_room",
                    choices=['cramped_room', 'coordination_ring', 'asymmetric_advantages_tomato', 'bonus_order_test', 'corridor', 'multiplayer_schelling'],
                    help="Choice for overcooked layout.")
args = parser.parse_args()

print(base_layout_params(args.layout, 400))

env = OvercookedMadrona(args.layout, args.num_envs, 0, args.debug_compile, args.use_cpu, args.use_env_cpu)

# old_state = env.n_reset()
actions = torch.zeros((2, args.num_envs, 1), dtype=int).to(device=env.device)
num_errors = 0

# warp up
for _ in range(5):
    torch.randint(high=env.action_space.n, size=actions.size(), out=actions)
    next_state, reward, next_done, _ = env.n_step(actions)


action_slice = env.static_actions[:]

time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    torch.randint(high=env.action_space.n, size=actions.size(), out=action_slice)

    time_stamps[iter * 2] = time.time()
    env.sim.step()
    time_stamps[iter * 2 + 1] = time.time()

time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
