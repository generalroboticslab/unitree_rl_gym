from sdl_gym.envs.g1_knob.g1_knob_4.g1_robot_config_4 import G1RobotCfg, G1RobotCfgPPO

class G1KnobCfg_4( G1RobotCfg ):
    class init_state_g1( G1RobotCfg.init_state_g1 ):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            
            # 6 DOFs of left leg PLUS foot 
            'left_hip_pitch_joint' : 0,  # index: 0
            'left_hip_roll_joint' : 0,      # index: 1
            'left_hip_yaw_joint' : 0.,      # index: 2
            'left_knee_joint' : 0.,        # index: 3
            'left_ankle_pitch_joint' : 0., # index: 4
            'left_ankle_roll_joint' : 0,    # index: 5

            # 6 DOFs of right leg PLUS foot   
            'right_hip_pitch_joint' : 0, # index: 6
            'right_hip_roll_joint' : 0,     # index: 7
            'right_hip_yaw_joint' : 0.,     # index: 8
            'right_knee_joint' : 0.,       # index: 9
            'right_ankle_pitch_joint': 0., # index: 10
            'right_ankle_roll_joint' : 0,   # index: 11

            # 7 DOFs of left arm
            'left_shoulder_pitch_joint' : 0., # index: 12
            'left_shoulder_roll_joint' : 0.,  # index: 13
            'left_shoulder_yaw_joint' : 0.,   # index: 14
            'left_elbow_joint' : 0.,          # index: 15
            'left_wrist_roll_joint' : 0.,     # index: 16
            'left_wrist_pitch_joint' : 0.,    # index: 17
            'left_wrist_yaw_joint' : 0.,      # index: 18

            # 7 DOFs of left hand
            'left_hand_index_0_joint' : 0.,   # index: 19
            'left_hand_index_1_joint' : 0.,   # index: 20
            'left_hand_middle_0_joint' : 0.,  # index: 21
            'left_hand_middle_1_joint' : 0.,  # index: 22
            'left_hand_thumb_0_joint' : 0.,   # index: 23
            'left_hand_thumb_1_joint' : 0.,   # index: 24
            'left_hand_thumb_2_joint' : 0.,   # index: 25

            # 7 DOFs of right arm
            'right_shoulder_pitch_joint' : 0., # index: 26
            'right_shoulder_roll_joint' : 0.,  # index: 27
            'right_shoulder_yaw_joint' : 0.,   # index: 28
            'right_elbow_joint' : 0.,          # index: 29
            'right_wrist_roll_joint' : 0.,     # index: 30
            'right_wrist_pitch_joint' : 0.,    # index: 31
            'right_wrist_yaw_joint' : 0.,      # index: 32

            # 7 DOFs of right hand
            'right_hand_index_0_joint' : 0.,   # index: 33
            'right_hand_index_1_joint' : 0.,   # index: 34
            'right_hand_middle_0_joint' : 0.,  # index: 35
            'right_hand_middle_1_joint' : 0.,  # index: 36
            'right_hand_thumb_0_joint' : 0.,   # index: 37
            'right_hand_thumb_1_joint' : 0.,   # index: 38
            'right_hand_thumb_2_joint' : 0.,   # index: 39
        }

        
    class init_state_knob( G1RobotCfg.init_state_knob ):
        pass
        
    
    class env(G1RobotCfg.env):
        num_observations = 101
        num_privileged_obs = 104
        num_actions = 13 # 6 + 7 (6 for the hand pos&rot, 7 for the fingers' DOFs)


    class domain_rand(G1RobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( G1RobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P' # P: position, V: velocity, T: torques
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 40,
                     
                     'hand': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 2,
                     
                     'hand': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset_g1( G1RobotCfg.asset_g1 ):
        file = '{SDL_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_with_hand_lock_waist.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        use_mesh_materials = True
        default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        
    class asset_knob (G1RobotCfg.asset_knob):
        file = '{SDL_GYM_ROOT_DIR}/resources/task_assets/knob/knob.urdf'
        name = 'knob'
        default_dof_drive_mode = 0
  
    class rewards( G1RobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( G1RobotCfg.rewards.scales ):
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.00001
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0

class G1KnobCfgPPO_4( G1RobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( G1RobotCfgPPO.algorithm ):
        pass
    class runner( G1RobotCfgPPO.runner ):
        # policy_class_name = "ActorCriticRecurrent"
        policy_class_name = "ActorCritic"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_knob'

  
