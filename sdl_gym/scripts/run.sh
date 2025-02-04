# Activate the Python environment if needed

# Knob Task

############### These 5 are IK implementation ###############
# python train.py --task=g1_knob_1 --num_envs=64
# python train.py --task=g1_knob_2 --num_envs=64
# python train.py --task=g1_knob_3 --num_envs=64
# python train.py --task=g1_knob_4 --num_envs=64
# python train.py --task=g1_knob_5 --num_envs=64
#############################################################


# Policy-Basic-0
# Train
# python train.py --task=g1_knob_6 --num_envs=4096
# Play
# python play.py --task=g1_knob_6 --num_envs=64 --load_run='Feb04_13-11-34_'


# Policy-1
# Train
python train.py --task=g1_knob_7 --num_envs=4096
# Play
# python play.py --task=g1_knob_7 --num_envs=64 --load_run='Feb04_13-11-34_'







# DEBUG
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client play.py --task=g1_knob_6 --num_envs=2 --headless