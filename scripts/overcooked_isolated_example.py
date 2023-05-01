from envs.overcooked_env import get_base_layout_params, OvercookedMadrona


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
parser.add_argument("--layout", type=str, default="cramped_room",
                    # choices=['cramped_room', 'coordination_ring', 'asymmetric_advantages_tomato', 'bonus_order_test', 'corridor', 'multiplayer_schelling'],
                    help="Choice for overcooked layout.")

parser.add_argument("--num-players", type=int, default=None,
                    help="the number of players in the game")
args = parser.parse_args()

print(get_base_layout_params(args.layout, 400))

env = OvercookedMadrona(args.layout, args.num_envs, 0, args.debug_compile, args.use_cpu, args.use_env_cpu, num_players=args.num_players)

chosen_actions = torch.zeros_like(env.static_actions).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_dones = torch.zeros_like(env.static_dones).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_observations = torch.zeros_like(env.static_observations).to('cuda' if torch.cuda.is_available() else 'cpu')
gpu_rewards = torch.zeros_like(env.static_rewards).to('cuda' if torch.cuda.is_available() else 'cpu')

# warp up
for _ in range(5):
    torch.randint(high=env.action_space.n, size=chosen_actions.size(), out=chosen_actions)

    env.static_actions.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_rewards.copy_(env.static_rewards)


action_slice = env.static_actions[:]

time_stamps = [0 for i in range(args.num_steps * 2)]
for iter in tqdm(range(args.num_steps), desc="Running Simulation"):
    torch.randint(high=env.action_space.n, size=chosen_actions.size(), out=chosen_actions)

    time_stamps[iter * 2] = time.time()
    env.static_actions.copy_(chosen_actions)
    env.sim.step()
    gpu_dones.copy_(env.static_dones)
    gpu_observations.copy_(env.static_observations)
    gpu_rewards.copy_(env.static_rewards)
    time_stamps[iter * 2 + 1] = time.time()

time_difference = [time_stamps[i] - time_stamps[i-1] for i in range(1, len(time_stamps), 2)]
assert(len(time_difference) == args.num_steps)
print("step * worlds / sec:", args.num_envs / (sum(time_difference) / args.num_steps))
