import torch

from envs.overcooked2_env import SimplifiedOvercooked, OvercookedMadrona
from envs.balance_beam_env import BalanceMadronaTorch, PantheonLine

from pantheonrl_extension.vectorenv import SyncVectorEnv

def generate_env(name, num_envs, layout='simple', use_baseline=False):
    if name == 'balance':
        if use_baseline:
            return SyncVectorEnv(
                [lambda: PantheonLine() for _ in range(num_envs)]
            )
        else:
            return BalanceMadronaTorch(num_envs, 0, debug_compile=False)
    elif name == 'overcooked':
        if use_baseline:
            return SyncVectorEnv(
                [lambda: SimplifiedOvercooked(layout) for _ in range(num_envs)]
            )
        else:
            return OvercookedMadrona(layout, num_envs, 0, debug_compile=False)
    else:
        raise Exception("Invalid environment name")
