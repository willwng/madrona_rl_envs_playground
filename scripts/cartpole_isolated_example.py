from envs.cartpole_env import CartpoleMadronaTorch
import torch
import time

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32,
                    help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=1000,
                    help="the number of steps to run in each environment per policy rollout")

parser.add_argument("--use-cpu", default=False, nargs="?", const=True,
                    help="if toggled, use cpu version of madrona")
parser.add_argument("--use-env-cpu", default=False, nargs="?", const=True,
                    help="if toggled, use cpu for env outputs")

parser.add_argument("--debug-compile", default=False, nargs="?", const=True,
                    help="if toggled, validate correctness")
args = parser.parse_args()

env = CartpoleMadronaTorch(args.num_envs, 0, args.debug_compile, args.use_cpu, args.use_env_cpu)
actions = torch.zeros((args.num_envs, 1), dtype=int).to(device=env.device)

num_errors = 0
start_time = time.time()

chosen_actions = torch.zeros_like(env.static_actions).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_dones = torch.zeros_like(env.static_dones).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_observations = torch.zeros_like(env.static_observations).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_rewards = torch.zeros_like(env.static_rewards).to('cuda' if torch.cuda.is_available() else 'cpu')

action_slice = env.static_actions[:]

for _ in range(5):
    torch.randint(high=2, size=actions.size(), out=action_slice)

    action_slice.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_rewards.copy_(env.static_rewards)

time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    torch.randint(high=2, size=actions.size(), out=chosen_actions)

    time_stamps[iter * 2] = time.time()
    action_slice.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_rewards.copy_(env.static_rewards)
    time_stamps[iter * 2 + 1] = time.time()

time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
