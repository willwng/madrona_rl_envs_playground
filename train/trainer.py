from MAPPO.main_player import MainPlayer

from config import get_config
import os
from pathlib import Path

from env_utils import generate_env

from partner_agents import CentralizedAgent

args = get_config().parse_args()
# args.hanabi_name = 'MaskedHanabi'
# args.n_rollout_threads = 1
# args.episode_length = 2000
# args.ppo_epoch = 15
# args.gain = 0.01
# args.lr = 7e-4
# args.critic_lr = 1e-3
# args.hidden_size = 512
# args.layer_N = 2
# args.entropy_coef = 0.015
# args.use_recurrent_policy = False
# args.use_value_active_masks = False
# args.use_policy_active_masks = False
# print(args)

# han_config={
#             "colors":
#                 2,
#             "ranks":
#                 5,
#             "players":
#                 2,
#             "hand_size":
#                 2,
#             "max_information_tokens":
#                 3,
#             "max_life_tokens":
#                 1,
#             "observation_type":1
#         }
# env = MaskedHanabi(han_config)
# print(env.observation_space)
# print(env.share_observation_space)
# print(env.action_space)

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout)

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
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))
config = {
    'all_args': args,
    'envs': envs,
    'device': 'cpu',
    'num_agents': 2,
    'run_dir': Path(run_dir)
}
ego = MainPlayer(config)
partner = CentralizedAgent(ego, 1)
envs.add_partner_agent(partner)
ego.run()
