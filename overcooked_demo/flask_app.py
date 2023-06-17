import os
import io
import json
import copy
import argparse
import torch
import numpy as np
import gym

import flask
from flask import Flask, jsonify, request
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
# from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

# from overcooked_utils import NAME_TRANSLATION

# from overcooked_env import DecentralizedOvercooked

# from partner_agents import DecentralizedAgent
# from MAPPO.r_actor_critic import R_Actor
# from config import get_config

app = Flask(__name__)


@app.route('/')
def root():
    return flask.send_file('index.html')


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
