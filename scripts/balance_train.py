import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.balance_beam_env import BalanceMadronaTorch, PantheonLine
from pantheonrl_extension.vectoragent import RandomVectorAgent, CleanPPOAgent
from pantheonrl_extension.vectorenv import SyncVectorEnv

from tqdm import tqdm


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--madrona", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, madrona will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=120,
        help="the number of parallel game environments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=60,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args

def make_env(seed, idx):
    def thunk():
        env = PantheonLine()
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

args = parse_args()
run_name = f"Balance_Train__{args.seed}__{int(time.time())}__{args.num_envs}_{'madrona' if args.madrona else 'numpy'}"

print("Run saved to:", run_name)

if args.madrona:
    env = BalanceMadronaTorch(args.num_envs, 0, False)
else:
    env = SyncVectorEnv(
            [make_env(args.seed + i, i) for i in range(args.num_envs)]
        )
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

assert args.cuda

num_updates = args.total_timesteps // int(args.num_envs * args.num_steps)

ego = CleanPPOAgent(
    envs = env,
    name = run_name + "_ego",
    device = torch.device("cpu"),
    num_updates = num_updates,
    verbose = True,
    lr = args.learning_rate,
    num_steps = args.num_steps,
    anneal_lr = args.anneal_lr,
    gamma = args.gamma,
    gae_lambda = args.gae_lambda,
    num_minibatches = args.num_minibatches,
    update_epochs = args.update_epochs,
    norm_adv = args.norm_adv,
    clip_coef = args.clip_coef,
    clip_vloss = args.clip_vloss,
    ent_coef = args.ent_coef,
    vf_coef = args.vf_coef,
    max_grad_norm = args.max_grad_norm,
    target_kl = args.target_kl
)

partner = CleanPPOAgent(
    envs = env,
    name = run_name + "_alt",
    device = torch.device("cpu"),
    num_updates = num_updates,
    verbose = True,
    lr = args.learning_rate,
    num_steps = args.num_steps,
    anneal_lr = args.anneal_lr,
    gamma = args.gamma,
    gae_lambda = args.gae_lambda,
    num_minibatches = args.num_minibatches,
    update_epochs = args.update_epochs,
    norm_adv = args.norm_adv,
    clip_coef = args.clip_coef,
    clip_vloss = args.clip_vloss,
    ent_coef = args.ent_coef,
    vf_coef = args.vf_coef,
    max_grad_norm = args.max_grad_norm,
    target_kl = args.target_kl
)

# partner = RandomVectorAgent(lambda: torch.randint_like(env.static_actions[0], high=4))

env.add_partner_agent(partner)
obs = env.reset()
for iter in tqdm(range(num_updates * args.num_steps), desc="Training Agent"):
    action = ego.get_action(obs, record=True)

    obs, rewards, dones, _ = env.step(action)

    ego.update(rewards, dones)
