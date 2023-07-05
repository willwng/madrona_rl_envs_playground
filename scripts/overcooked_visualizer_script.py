from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from collections import namedtuple

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=100,
                    help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=100,
                    help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--out-dir", default="overcooked_output",
                    help="directory to save overcooked frames")
args = parser.parse_args()

class SimplifiedState(namedtuple('SimplifiedState', 'players objects')):
    pass


ae = AgentEvaluator.from_layout_name({"layout_name": "cramped_room"}, {"horizon": args.num_steps})
trajs = {}
trajs['mdp_params'] = [{'layout_name': 'cramped_room', 'terrain': [['X', 'X', 'P', 'X', 'X'], ['O', ' ', ' ', ' ', 'O'], ['X', ' ', ' ', ' ', 'X'], ['X', 'D', 'X', 'S', 'X']], 'start_player_positions': [(1, 2), (3, 1)], 'start_bonus_orders': [], 'rew_shaping_params': {'PLACEMENT_IN_POT_REW': 3, 'DISH_PICKUP_REWARD': 0, 'SOUP_PICKUP_REWARD': 5, 'DISH_DISP_DISTANCE_REW': 0, 'POT_DISTANCE_REW': 0, 'SOUP_DISTANCE_REW': 0}, 'start_all_orders': [{'ingredients': ['onion', 'onion', 'onion']}]}]

for i in range(args.num_envs):
    temp_trajs = ae.evaluate_random_pair()

    states = temp_trajs["ep_states"][0]
    simplified_states = []
    for state in states:
        simplified_states.append(SimplifiedState(state.players, state.objects))

    trajs['ep_states'] = [simplified_states]

    StateVisualizer(is_rendering_hud=False, is_rendering_cooking_timer=False, tile_size=15).display_rendered_trajectory(trajs, img_directory_path=f"{args.out_dir}/{i}", ipython_display=False)
