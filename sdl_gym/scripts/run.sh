# Activate the Python environment if needed

# Knob Task
# Train
# python train.py --task=g1_knob_1 --num_envs=64
# python train.py --task=g1_knob_2 --num_envs=64
# python train.py --task=g1_knob_3 --num_envs=64
# python train.py --task=g1_knob_4 --num_envs=64
# python train.py --task=g1_knob_5 --num_envs=64

# python train.py --task=g1_knob_6 --num_envs=4096 --headless
# python train.py --task=g1_knob_6 --num_envs=4096
python play.py --task=g1_knob_6 --num_envs=64
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client play.py --task=g1_knob_6 --num_envs=2 --run_name='/home/grl/isaacgym/self_driven_lab/unitree_rl_gym/logs/g1_knob/Feb03_15-57-51_/model_300.pt' 