from sdl_gym.envs.g1_knob.g1_robot_config import G1RobotCfg, G1RobotCfgPPO

class G1KnobCfg( G1RobotCfg ):
    class init_state_g1( G1RobotCfg.init_state_g1 ):
        default_joint_angles = { # = target angles [rad] when action = 0.0
        
        # 6 DOFs of left leg PLUS foot                        
        'left_hip_yaw_joint' : 0. ,   
        'left_hip_roll_joint' : 0,               
        'left_hip_pitch_joint' : -0.1,         
        'left_knee_joint' : 0.3,       
        'left_ankle_pitch_joint' : -0.2,     
        'left_ankle_roll_joint' : 0,    
            
        # 6 DOFs of right leg PLUS foot   
        'right_hip_yaw_joint' : 0., 
        'right_hip_roll_joint' : 0, 
        'right_hip_pitch_joint' : -0.1,                                       
        'right_knee_joint' : 0.3,                                             
        'right_ankle_pitch_joint': -0.2,                              
        'right_ankle_roll_joint' : 0,       
        
        # 3 DOFs of waist
        'waist_yaw_joint' : 0.,
        'waist_roll_joint' : 0.,
        'waist_pitch_joint' : 0.,
           
        # 7 DOFs of left arm
        'left_shoulder_pitch_joint' : 0.,
        'left_shoulder_roll_joint' : 0.,
        'left_shoulder_yaw_joint' : 0.,
        'left_elbow_joint' : 0.,
        'left_wrist_roll_joint' : 0.,
        'left_wrist_pitch_joint' : 0.,
        'left_wrist_yaw_joint' : 0.,
              
        # 7 DOFs of left hand
        'left_hand_thumb_0_joint' : 0.,
        'left_hand_thumb_1_joint' : 0.,
        'left_hand_thumb_2_joint' : 0.,
        'left_hand_middle_0_joint' : 0.,
        'left_hand_middle_1_joint' : 0.,
        'left_hand_index_0_joint' : 0.,
        'left_hand_index_1_joint' : 0.,
        
        # 7 DOFs of right arm
        'right_shoulder_pitch_joint' : 0.,
        'right_shoulder_roll_joint' : 0.,
        'right_shoulder_yaw_joint' : 0.,
        'right_elbow_joint' : 0.,
        'right_wrist_roll_joint' : 0.,
        'right_wrist_pitch_joint' : 0.,
        'right_wrist_yaw_joint' : 0.,
        
        # 7 DOFs of right hand
        'right_hand_thumb_0_joint' : 0.,
        'right_hand_thumb_1_joint' : 0.,
        'right_hand_thumb_2_joint' : 0.,
        'right_hand_middle_0_joint' : 0.,
        'right_hand_middle_1_joint' : 0.,
        'right_hand_index_0_joint' : 0.,
        'right_hand_index_1_joint' : 0.,
        
        #    'torso_joint' : 0.
        
        }
        
    class init_state_knob( G1RobotCfg.init_state_knob ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        
    
    class env(G1RobotCfg.env):
        num_observations = 140
        num_privileged_obs = 143
        num_actions = 43


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
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     
                     'waist': 100,
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
                     
                     'waist': 2,
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
        file = '{SDL_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_with_hand.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class asset_knob (G1RobotCfg.asset_knob):
        file = '{SDL_GYM_ROOT_DIR}/resources/task_assets/knob/knob.urdf'
        name = 'knob'
        default_dof_drive_mode = 0
  
    class rewards( G1RobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( G1RobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0

class G1KnobCfgPPO( G1RobotCfgPPO ):
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
        entropy_coef = 0.01
    class runner( G1RobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_knob'

  
