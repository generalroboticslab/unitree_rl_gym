from sdl_gym import SDL_GYM_ROOT_DIR, SDL_GYM_ENVS_DIR

from sdl_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from sdl_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from sdl_gym.envs.h1.h1_env import H1Robot
from sdl_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from sdl_gym.envs.h1_2.h1_2_env import H1_2Robot
from sdl_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from sdl_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from sdl_gym.envs.g1_knob.g1_knob_config import G1KnobCfg, G1KnobCfgPPO
from sdl_gym.envs.g1_knob.g1_knob_env import G1KnobRobot

from sdl_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_knob", G1KnobRobot, G1KnobCfg(), G1KnobCfgPPO())