from MAPPO.main_player import MainPlayer

from config import get_config
import os
from pathlib import Path

from env_utils import generate_env

from partner_agents import CentralizedAgent

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Policy(nn.Module):

    def __init__(self, actor):
        super(Policy, self).__init__()

        self.base = actor.base.cnn.cnn
        self.act_layer = actor.act

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float)
        x = self.base(x.permute((0, 3, 1, 2)))
        x = self.act_layer(x, deterministic=True)
        return x[0]


args = get_config().parse_args()

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout, use_env_cpu=(device=='cpu'), use_baseline=args.use_baseline)

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
    os.path.dirname(os.path.abspath(__file__))
    + "/"
    + args.hanabi_name
    + "/results/"
    + (args.run_dir)
    + "/"
    + str(args.seed)
)

run_dir = Path(run_dir)
args.model_dir = str(run_dir / 'models')

config = {
    'all_args': args,
    'envs': envs,
    'device': device,
    'num_agents': 2,
    'run_dir': run_dir
}
ego = MainPlayer(config)
torch_network = Policy(ego.policy.actor)

actions = torch.zeros((2, args.n_rollout_threads, 1), dtype=int, device=device)

state1, state2 = envs.n_reset()
scores = torch.zeros(args.n_rollout_threads, device=device)
for i in range(args.env_length):
    actions[0, :, :] = torch_network(state1.obs)
    actions[1, :, :] = torch_network(state2.obs)
    (state1, state2), reward, _, _ = envs.n_step(actions)
    scores += reward[0, :]
score_vals, counts = torch.unique(scores, return_counts=True)
print({x.item() : y.item() for x, y in zip(score_vals, counts)})
