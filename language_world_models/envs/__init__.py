import sys
import inspect
import random

from marlgrid.base import MultiGridEnv
from marlgrid.agents import GridAgentInterface
from .choosepath import ChoosePathGrid

this_module = sys.modules[__name__]

def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k:v for k,v in globals().items() if inspect.isclass(v) and issubclass(v, MultiGridEnv)}
    
    env_class = possible_envs[env_config['env_class']]
    
    env_kwargs = {k:v for k,v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337*1337)
    
    return env_class(**env_kwargs)


def create_ChoosePathGridDefaultEnv():
    env_config =  {
        "env_class": "ChoosePathGrid",
        "grid_size": 11,
        "max_steps": 30,
        "respawn": False,
        "ghost_mode": True,
        "reward_decay": False,
    }

    player_interface_config = {
        "view_size": 3,
        "view_offset": 1,
        "view_tile_size": 11,
        "observation_style": 'image',
        "see_through_walls": False,
        "color": "prestige"
    }

    # Add the player/agent config to the environment config (as expected by "env_from_config" below)
    env_config['agents'] = [player_interface_config]
    return env_from_config(env_config)
