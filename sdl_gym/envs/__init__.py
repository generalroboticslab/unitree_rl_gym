from sdl_gym import SDL_GYM_ROOT_DIR, SDL_GYM_ENVS_DIR

from sdl_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from sdl_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from sdl_gym.envs.g1_knob.g1_knob_config import G1KnobCfg, G1KnobCfgPPO
from sdl_gym.envs.g1_knob.g1_knob_env import G1KnobRobot
from sdl_gym.envs.g1_knob.g1_knob_env_ik_test import G1KnobRobotIKTest
from sdl_gym.envs.g1_knob.g1_knob_env_ik_test_onlyarmhand import G1KnobRobotIKTestOnlyarmhand

from sdl_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_knob", G1KnobRobot, G1KnobCfg(), G1KnobCfgPPO())
task_registry.register( "g1_knob_ik_test", G1KnobRobotIKTest, G1KnobCfg(), G1KnobCfgPPO())
task_registry.register( "g1_knob_ik_test_onlyarmhand", G1KnobRobotIKTestOnlyarmhand, G1KnobCfg(), G1KnobCfgPPO())