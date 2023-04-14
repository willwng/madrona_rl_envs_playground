from envs.cartpole_env import CartpoleMadronaTorch, validate_step, CartpoleNumpy
import torch
import time

import gym

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")

parser.add_argument("--use-cpu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, use cpu version of madrona")
parser.add_argument("--use-env-cpu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, use cpu for env outputs")

parser.add_argument("--debug-compile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, validate correctness")
args = parser.parse_args()

env = CartpoleMadronaTorch(args.num_envs, 0, args.debug_compile, args.use_cpu, args.use_env_cpu)
actions = torch.zeros((args.num_envs, 1), dtype=int).to(device=env.device)

old_state = env.reset()
num_errors = 0
start_time = time.time()

for _ in range(5):
    torch.randint(high=2, size=actions.size(), out=actions)
    
    next_state, reward, next_done, _ = env.step(actions)
    old_state = next_state

action_slice = env.static_actions[:]
    
time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    torch.randint(high=2, size=actions.size(), out=action_slice)

    time_stamps[iter * 2] = time.time()
    env.sim.step()
    time_stamps[iter * 2 + 1] = time.time()

time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
