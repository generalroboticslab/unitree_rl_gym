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
# python train.py --task=g1_knob_7 --num_envs=4096
# python train.py --task=g1_knob_7 --num_envs=4096 --resume --load_run='Feb12_22-47-10_'
# Play - Workable 1: Feb04_18-24-43_
# python play.py --task=g1_knob_7 --num_envs=128 --load_run='Feb13_15-15-01_'
# python play.py --task=g1_knob_7 --num_envs=128 --load_run='Feb12_22-47-10_' --checkpoint=700

# Policy-traditional
# Train
# python train.py --task=g1_knob_traditional --num_envs=4096
# python train.py --task=g1_knob_traditional --num_envs=4096 --resume --load_run='Feb12_22-47-10_'
# Play - Workable 1: Feb04_18-24-43_
python play.py --task=g1_knob_traditional --num_envs=64 --load_run='Feb13_14-28-28_'
# python play.py --task=g1_knob_traditional --num_envs=128 --load_run='Feb12_22-47-10_' --checkpoint=700

# DEBUG
# Train
# python train.py --task=g1_knob_debug --num_envs=4096 
# python train.py --task=g1_knob_debug --num_envs=4096 --resume --load_run='Feb11_18-31-04_'
# Play - Workable 1: Feb04_18-24-43_
# python play.py --task=g1_knob_debug --num_envs=128 --load_run='Feb11_16-18-46_' --checkpoint=500
# python play.py --task=g1_knob_debug --num_envs=128 --load_run='Feb11_22-37-43_'




# DEBUG
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client play.py --task=g1_knob_6 --num_envs=2 --headless