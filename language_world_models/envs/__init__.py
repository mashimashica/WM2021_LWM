from .choosepath import ChoosePathGrid
from marlgrid.base import MultiGridEnv
from marlgrid.agents import GridAgentInterface
from gym.envs.registration import register as gym_register

import sys
import inspect
import random

this_module = sys.modules[__name__]
registered_envs = []


def register_marl_env(
    env_name,
    env_class,
    n_agents,
    grid_size,
    view_size,
    view_tile_size=8,
    view_offset=0,
    observation_style='image',
    see_through_walls=False,
    agent_color=None,
    env_kwargs={},
):
    colors = ["red", "blue", "purple", "orange", "olive", "pink"]
    assert n_agents <= len(colors)

    class RegEnv(env_class):
        def __new__(cls):
            instance = super(env_class, RegEnv).__new__(env_class)
            instance.__init__(
                agents=[
                    GridAgentInterface(
                        color=c if agent_color is None else agent_color,
                        view_size=view_size,
                        view_tile_size=view_tile_size,
                        view_offset=view_offset,
                        observation_style=observation_style,
                        see_through_walls=see_through_walls,
                        )
                    for c in colors[:n_agents]
                ],
                grid_size=grid_size,
                **env_kwargs,
            )
            return instance

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"marlgrid.envs:{env_class_name}")


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k:v for k,v in globals().items() if inspect.isclass(v) and issubclass(v, MultiGridEnv)}
    
    env_class = possible_envs[env_config['env_class']]
    
    env_kwargs = {k:v for k,v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337*1337)
    
    return env_class(**env_kwargs)


register_marl_env(
    "ChoosePathGrid9x9-v0",
    ChoosePathGrid,
    n_agents=1,
    grid_size=11,
    view_size=3,
    view_tile_size=11,
    view_offset=1,
    observation_style="rich",
    see_through_walls=False,
    env_kwargs={
        'max_steps' : 30,
        'respawn' : False,
    }
)

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
