#!/bin/bash

# Activate the Python environment if needed



# Train
# python train.py --task=g1_knob --num_envs=64
python train.py --task=g1_knob_ik_test_onlyarmhand --num_envs=64
# python train.py --task=g1_knob_ik_test_onlyarmhand --num_envs=64 --headless
# python train.py --task=g1_knob_ik_test --num_envs=64



# Inference

