from sdl_gym import SDL_GYM_ROOT_DIR, SDL_GYM_ENVS_DIR
from sdl_gym.utils.task_registry import task_registry

from sdl_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from sdl_gym.envs.g1.g1_env import G1Robot

# Knob
# 1
from sdl_gym.envs.g1_knob.g1_knob_1.g1_knob_config_1 import G1KnobCfg_1, G1KnobCfgPPO_1
from sdl_gym.envs.g1_knob.g1_knob_1.g1_knob_env_1 import G1KnobRobot_1

# 2
from sdl_gym.envs.g1_knob.g1_knob_2.g1_knob_config_2 import G1KnobCfg_2, G1KnobCfgPPO_2
from sdl_gym.envs.g1_knob.g1_knob_2.g1_knob_env_2 import G1KnobRobot_2

# 3
from sdl_gym.envs.g1_knob.g1_knob_3.g1_knob_config_3 import G1KnobCfg_3, G1KnobCfgPPO_3
from sdl_gym.envs.g1_knob.g1_knob_3.g1_knob_env_3 import G1KnobRobot_3

# 4
from sdl_gym.envs.g1_knob.g1_knob_4.g1_knob_config_4 import G1KnobCfg_4, G1KnobCfgPPO_4
from sdl_gym.envs.g1_knob.g1_knob_4.g1_knob_env_4 import G1KnobRobot_4

# 5
from sdl_gym.envs.g1_knob.g1_knob_5.g1_knob_config_5 import G1KnobCfg_5, G1KnobCfgPPO_5
from sdl_gym.envs.g1_knob.g1_knob_5.g1_knob_env_5 import G1KnobRobot_5

# 6
from sdl_gym.envs.g1_knob.g1_knob_6.g1_knob_config_6 import G1KnobCfg_6, G1KnobCfgPPO_6
from sdl_gym.envs.g1_knob.g1_knob_6.g1_knob_env_6 import G1KnobRobot_6

# 7
from sdl_gym.envs.g1_knob.g1_knob_7.g1_knob_config_7 import G1KnobCfg_7, G1KnobCfgPPO_7
from sdl_gym.envs.g1_knob.g1_knob_7.g1_knob_env_7 import G1KnobRobot_7

# traditional
from sdl_gym.envs.g1_knob.g1_knob_traditional.g1_knob_config_traditional import G1KnobCfg_traditional, G1KnobCfgPPO_traditional
from sdl_gym.envs.g1_knob.g1_knob_traditional.g1_knob_env_traditional import G1KnobRobot_traditional

# traditional playboard
from sdl_gym.envs.g1_knob.g1_knob_traditional_playboard.g1_knob_config_traditional_playboard import G1KnobCfg_traditional_playboard, G1KnobCfgPPO_traditional_playboard
from sdl_gym.envs.g1_knob.g1_knob_traditional_playboard.g1_knob_env_traditional_playboard import G1KnobRobot_traditional_playboard

# rl playboard
from sdl_gym.envs.g1_knob.g1_knob_rl_playboard.g1_knob_config_rl_playboard import G1KnobCfg_rl_playboard, G1KnobCfgPPO_rl_playboard
from sdl_gym.envs.g1_knob.g1_knob_rl_playboard.g1_knob_env_rl_playboard import G1KnobRobot_rl_playboard

# debug
from sdl_gym.envs.g1_knob.g1_knob_debug.g1_knob_config_debug import G1KnobCfg_debug, G1KnobCfgPPO_debug
from sdl_gym.envs.g1_knob.g1_knob_debug.g1_knob_env_debug import G1KnobRobot_debug

task_registry.register( "g1_knob_1", G1KnobRobot_1, G1KnobCfg_1(), G1KnobCfgPPO_1())
task_registry.register( "g1_knob_2", G1KnobRobot_2, G1KnobCfg_2(), G1KnobCfgPPO_2())
task_registry.register( "g1_knob_3", G1KnobRobot_3, G1KnobCfg_3(), G1KnobCfgPPO_3())
task_registry.register( "g1_knob_4", G1KnobRobot_4, G1KnobCfg_4(), G1KnobCfgPPO_4())
task_registry.register( "g1_knob_5", G1KnobRobot_5, G1KnobCfg_5(), G1KnobCfgPPO_5())
task_registry.register( "g1_knob_6", G1KnobRobot_6, G1KnobCfg_6(), G1KnobCfgPPO_6())
task_registry.register( "g1_knob_7", G1KnobRobot_7, G1KnobCfg_7(), G1KnobCfgPPO_7())
task_registry.register( "g1_knob_traditional", G1KnobRobot_traditional, G1KnobCfg_traditional(), G1KnobCfgPPO_traditional())
task_registry.register( "g1_knob_debug", G1KnobRobot_debug, G1KnobCfg_debug(), G1KnobCfgPPO_debug())
task_registry.register( "g1_knob_traditional_playboard", G1KnobRobot_traditional_playboard, G1KnobCfg_traditional_playboard(), G1KnobCfgPPO_traditional_playboard())
task_registry.register( "g1_knob_rl_playboard", G1KnobRobot_rl_playboard, G1KnobCfg_rl_playboard(), G1KnobCfgPPO_rl_playboard())



# G1 original locomotion
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())