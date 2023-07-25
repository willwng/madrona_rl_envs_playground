from gym.vector.vector_env import VectorEnv
import build.madrona_python as madrona_python
import build.madrona_cartpole_example_python as cartpole_python
import torch

from gym import logger, spaces

import numpy as np

from math import pi

import math
from typing import Optional, Union

import numpy as np

import gym
from gym.utils import seeding

def to_np(a):
    return a.cpu().numpy()


X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * pi / 360


class CartpoleMadronaNumpy(VectorEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, use_cpu=False, use_env_cpu=False):
        high = np.array(
            [
                X_THRESHOLD * 2,
                np.finfo(np.float32).max,
                THETA_THRESHOLD_RADIANS * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        if use_env_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(-high, high, dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.sim = cartpole_python.CartpoleSimulator(
            exec_mode = cartpole_python.ExecMode.CPU if use_cpu else cartpole_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
        )

        self.static_dones = self.sim.reset_tensor().to_torch()
        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()

        self.infos = [{}] * self.num_envs

    def step(self, actions):
        actions = actions[:, np.newaxis]
        self.static_actions.copy_(torch.from_numpy(actions))

        self.sim.step()

        return to_np(self.static_observations), to_np(self.static_rewards), to_np(self.static_dones[:, 0]), [{}] * self.num_envs

    def reset(self):
        return to_np(self.static_observations)

    def close(self, **kwargs):
        pass

class CartpoleMadronaTorch(VectorEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, use_cpu=False, use_env_cpu=False):
        high = np.array(
            [
                X_THRESHOLD * 2,
                np.finfo(np.float32).max,
                THETA_THRESHOLD_RADIANS * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        if use_env_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(-high, high, dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.sim = cartpole_python.CartpoleSimulator(
            exec_mode = cartpole_python.ExecMode.CPU if use_cpu else cartpole_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
        )

        self.static_dones = self.sim.reset_tensor().to_torch()
        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()

        self.infos = [{}] * self.num_envs

    def to_torch(self, a):
        return a.to(self.device)

    def step(self, actions):
        actions = actions[:, np.newaxis]
        self.static_actions.copy_(actions)
        
        self.sim.step()

        return self.to_torch(self.static_observations), self.to_torch(self.static_rewards), self.to_torch(self.static_dones[:, 0]), self.infos

    def reset(self):
        return self.to_torch(self.static_observations)

    def close(self, **kwargs):
        pass

class CartpoleNumpy(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def close(self):
        pass


STATIC_ENV = CartpoleNumpy()
    
def validate_step(states, actions, dones, nextstates, verbose=True):
    numenvs = dones.size(0)

    states = states.cpu().numpy()
    actions = actions[:, 0].cpu().numpy()
    dones = dones.cpu().numpy()
    nextstates = nextstates.cpu().numpy()

    retval = True
    
    for i in range(numenvs):
        STATIC_ENV.steps_beyond_done = None
        STATIC_ENV.state = states[i]
        truenext, _, truedone, _ = STATIC_ENV.step(actions[i])
        # if truedone:
        #     print("FINISHED EPISODE")
        
        if truedone != dones[i]:
            if verbose:
                print("start state:", states[i], i)
                print("action:", actions[i])
                print("madrona transition:", nextstates[i])
                print("numpy transition:", truenext)
                print(f"DONES mismatch: numpy={truedone}, madrona={dones[i] == 1}")
            retval = False
            # return False
            # pass
        if dones[i]:
            # print("MADRONA DONE", nextstates[i])
            continue

        if not np.all(np.abs(truenext - nextstates[i]) < 1e-6):
            if verbose:
                print("start state:", states[i], i)
                print("action:", actions[i])
                print("madrona transition:", nextstates[i])
                print("numpy transition:", truenext)
                print("TRANSITIONS are not equal")
            retval = False
            # return False
            # pass
    # print("All good")
    return retval
