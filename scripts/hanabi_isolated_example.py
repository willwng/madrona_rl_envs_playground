from envs.hanabi_env import HanabiMadrona, config_choice

from pantheonrl_extension.vectorobservation import VectorObservation

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
                    help="if toggled, use debug compilation mode")
parser.add_argument("--hanabi-type", type=str, default="full",
                    choices=['very_small', 'small', 'full'],
                    help="Choice for hanabi type.")
args = parser.parse_args()

han_conf = config_choice[args.hanabi_type]

env = HanabiMadrona(args.num_envs, 0, args.debug_compile, han_conf, args.use_cpu, args.use_env_cpu)
old_state = env.n_reset()
actions = torch.zeros((2, args.num_envs, 1), dtype=int).to(device=env.device)
num_errors = 0

chosen_actions = torch.zeros_like(env.static_actions).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_dones = torch.zeros_like(env.static_dones).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_observations = torch.zeros_like(env.static_observations).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_agent_states = torch.zeros_like(env.static_agent_states).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_action_masks = torch.zeros_like(env.static_action_masks).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_rewards = torch.zeros_like(env.static_rewards).to('cuda' if torch.cuda.is_available() else 'cpu')

# warp up
for _ in range(5):
    logits = torch.rand(2, args.num_envs, env.action_space.n).to(device=env.device)
    logits[torch.logical_not(env.static_action_masks[:, :, :env.discrete_action_size].to(torch.bool))] = -float('inf')
    chosen_actions[:, :, 0].copy_(torch.max(logits, dim=2).indices)

    env.static_actions.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_agent_states.copy_(env.static_agent_states)
    gpu_action_masks.copy_(env.static_action_masks)
    gpu_rewards.copy_(env.static_rewards)


time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    logits = torch.rand(2, args.num_envs, env.action_space.n).to(device=env.device)
    logits[torch.logical_not(env.static_action_masks[:, :, :env.discrete_action_size].to(torch.bool))] = -float('inf')
    chosen_actions[:, :, 0].copy_(torch.max(logits, dim=2).indices)

    time_stamps[iter * 2] = time.time()
    env.static_actions.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_agent_states.copy_(env.static_agent_states)
    gpu_action_masks.copy_(env.static_action_masks)
    gpu_rewards.copy_(env.static_rewards)
    time_stamps[iter * 2 + 1] = time.time()

time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
