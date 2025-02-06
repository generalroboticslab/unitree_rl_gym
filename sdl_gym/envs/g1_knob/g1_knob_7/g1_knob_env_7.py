from sdl_gym.envs.g1_knob.g1_knob_7.g1_robot_7 import G1Robot
from sdl_gym import SDL_GYM_ROOT_DIR, envs, SDL_GYM_ENVS_DIR
import time
from warnings import WarningMessage
import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from torch import Tensor
from typing import Tuple, Dict
from sdl_gym import SDL_GYM_ROOT_DIR
from sdl_gym.envs.base.base_task import BaseTask
from sdl_gym.utils.math import wrap_to_pi
from sdl_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from sdl_gym.utils.helpers import class_to_dict, print_g1_dof_index
from .g1_robot_config_7 import G1RobotCfg
from ....utils import factory_control as fc
from datetime import datetime
import wandb


class G1KnobRobot_7(G1Robot):
    
    # Is called during super().__init__, right after _create_envs()
    def _init_data(self):
          
        # Store some indexes / handles
        self.right_hand_fingers_dofs_num = 7
        self.right_arm_dofs_num = 7
        
        self.torso_link_index = self.gym.find_asset_rigid_body_index(self.g1_asset, "torso_link")
        self.right_shoulder_pitch_link_handle = self.gym.find_asset_rigid_body_index(self.g1_asset, "right_shoulder_pitch_link") # 29
        self.right_wrist_yaw_link_index = self.gym.find_asset_rigid_body_index(self.g1_asset, "right_wrist_yaw_link") # 36
        self.knob_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_knob[0], "knob")
        self.right_hand_palm_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_palm_link") 
        # Force Sensor Link Handles
        self.right_hand_thumb_2_force_sensor_1_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_thumb_2_force_sensor_1_link")
        self.right_hand_thumb_2_force_sensor_2_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_thumb_2_force_sensor_2_link")
        self.right_hand_thumb_2_force_sensor_3_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_thumb_2_force_sensor_3_link")
        self.right_hand_index_1_force_sensor_1_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_index_1_force_sensor_1_link")
        self.right_hand_index_1_force_sensor_2_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_index_1_force_sensor_2_link")
        self.right_hand_index_1_force_sensor_3_link_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_index_1_force_sensor_3_link")
        
        self.waist_yaw_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "waist_yaw_joint") # 12
        self.waist_pitch_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "waist_pitch_joint") # 14
        self.right_shoulder_pitch_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "right_shoulder_pitch_joint") # 29
        self.right_wrist_yaw_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "right_wrist_yaw_joint") # 35
        self.right_hand_index_0_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_index_0_joint") # 36
        self.right_hand_index_1_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_index_1_joint") # 37
        self.right_hand_thumb_2_joint_handle = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles_g1[0], "right_hand_thumb_2_joint") # 42
        
        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        
        
        knob_pose = self.gym.get_rigid_transform(self.envs[0], self.knob_handle)
        local_knob_hand_reach_pose = gymapi.Transform()
        local_knob_hand_reach_pose.p = gymapi.Vec3(0.15, -0.03, -0.025)
        local_knob_hand_reach_pose.r =  gymapi.Quat.from_euler_zyx(0, 0, np.pi * 5 / 8)
        global_knob_hand_reach_pose = knob_pose * local_knob_hand_reach_pose
        self.global_knob_hand_reach_pos = to_torch([global_knob_hand_reach_pose.p.x, global_knob_hand_reach_pose.p.y,
                                                global_knob_hand_reach_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.global_knob_hand_reach_rot = to_torch([global_knob_hand_reach_pose.r.x, global_knob_hand_reach_pose.r.y,
                                                global_knob_hand_reach_pose.r.z, global_knob_hand_reach_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        self.global_hand_pose = gymapi.Transform()
        self.global_knob_hand_reach_pose = global_knob_hand_reach_pose
        
        
        local_knob_center_keypoint_pose = gymapi.Transform()
        local_knob_center_keypoint_pose.p = gymapi.Vec3(0.045, 0, 0.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        local_knob_center_keypoint_pose.r = gymapi.Quat(0, 0, 0, 1)
        global_knob_center_keypoint_pose = knob_pose * local_knob_center_keypoint_pose
        self.global_knob_center_keypoint_pos = to_torch([global_knob_center_keypoint_pose.p.x, global_knob_center_keypoint_pose.p.y,
                                                global_knob_center_keypoint_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.global_knob_center_keypoint_rot = to_torch([global_knob_center_keypoint_pose.r.x, global_knob_center_keypoint_pose.r.y,
                                                global_knob_center_keypoint_pose.r.z, global_knob_center_keypoint_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        self.global_knob_center_keypoint_pose = global_knob_center_keypoint_pose
        
                        
        # Visualization setup
        self.axes_geom = gymutil.AxesGeometry(0.2)
        self.sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=self.sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.08, 12, 12, sphere_pose, color=(1, 1, 0))

                
    # Is called after super().__init__
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "g1")
        net_contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # self.root_states_g1 = self.root_states[:, 0].contiguous()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor_tensor)
        self.dof_state_g1 = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof_g1]
        self.dof_pos_g1 = self.dof_state_g1[..., 0]
        self.dof_vel_g1 = self.dof_state_g1[..., 1]
        # self.base_quat = self.root_states_g1[:, 3:7]
        # self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        # self.base_pos = self.root_states_g1[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces_tensor).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.dof_state_knob = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof_g1:]
        self.dof_pos_knob = self.dof_state_knob[..., 0]
        self.dof_vel_knob = self.dof_state_knob[..., 1]
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        self.right_hand_jacobian = self.jacobian[:, self.right_hand_palm_link_handle - 1, :, self.right_shoulder_pitch_joint_handle:self.right_wrist_yaw_joint_handle+1] # Need to be inspected

        self.g1_right_hand_pos = self.rigid_body_states[:, self.right_hand_palm_link_handle, 0:3]
        self.g1_right_hand_quat = self.rigid_body_states[:, self.right_hand_palm_link_handle, 3:7]
        self.g1_right_hand_7_dofs_pos = self.dof_pos_g1[:, self.right_hand_index_0_joint_handle:self.right_hand_thumb_2_joint_handle+1]
        self.g1_right_hand_7_dofs_vel = self.dof_vel_g1[:, self.right_hand_index_0_joint_handle:self.right_hand_thumb_2_joint_handle+1]
        self.right_hand_thumb_2_force_sensor_3_pos = self.rigid_body_states[:, self.right_hand_thumb_2_force_sensor_3_link_handle, 0:3]
        self.right_hand_index_1_force_sensor_3_pos = self.rigid_body_states[:, self.right_hand_index_1_force_sensor_3_link_handle, 0:3]
        
        self._setup_reward_history_buf_and_WandB()        
        
        self._refresh_tensors()
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof_g1, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof_g1, dtype=torch.float, device=self.device, requires_grad=False)
        self.network_output_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.real_dof_control_pos_target = torch.zeros(self.num_envs, 14, dtype=torch.float, device=self.device, requires_grad=False)
        self.input_sim_actions_vectors = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_network_output_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_dof_vel = torch.zeros_like(self.dof_vel_g1)
        # self.last_root_vel = torch.zeros_like(self.root_states_g1[:, 7:13])
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states_g1[:, 7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states_g1[:, 10:13])
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.knob_target_angle = torch.tensor([4 * torch.pi], device=self.device) 
        self.knob_initial_angle = torch.tensor([0 * torch.pi], device=self.device) 
      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs_g1, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs_g1):
            name = self.dof_names_g1[i]
            angle = self.cfg.init_state_g1.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos
    
    
    def _setup_reward_history_buf_and_WandB(self):
        if self.cfg.env.if_log_wandb and not self.cfg.env.test:
            wandb.init(
                    project="G1-knob",
                    name=f"{datetime.now().strftime('%b%d_%H-%M-%S')}"
                )
                    
        # Buffers for reward history and WandB logging
        self.total_reward_buf = torch.zeros(self.num_envs, device=self.device)

        self.knob_rotate_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.contact_forces_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.fingertips_TO_knob_center_dist_reward_buf = torch.zeros(self.num_envs, device=self.device)

        self.last_knob_rotation_percentage = torch.zeros(self.num_envs, device=self.device)
        self.last_contact_forces = torch.zeros(self.num_envs, device=self.device)
        self.last_thumb_force_sensor_3_TO_knob_center_keypoint_dist = torch.zeros(self.num_envs, device=self.device)
        self.last_index_force_sensor_3_TO_knob_center_keypoint_dist = torch.zeros(self.num_envs, device=self.device)
        
            
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
    

    
    
    def _create_envs(self):
        self._acquire_g1_asset_and_setup()
        self._acquire_knob_asset_and_setup()
        
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, self.cfg.env.env_lower, self.cfg.env.env_upper, int(np.sqrt(self.num_envs)))
            
            ################### Can add the rigid shape props setup here ###################
            ################################################################################
            
            actor_handle_g1 = self.gym.create_actor(env_handle, self.g1_asset, self.start_pose_g1, self.cfg.asset_g1.name, i, self.cfg.asset_g1.self_collisions, 0)
            dof_props_g1 = self._process_dof_props_g1(self.dof_props_asset_g1, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle_g1, dof_props_g1)
            
            actor_handle_knob = self.gym.create_actor(env_handle, self.knob_asset, self.start_pose_knob, self.cfg.asset_knob.name, i, self.cfg.asset_knob.self_collisions, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle_knob, self.dof_props_asset_knob)
            
            # Set knob colors
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8039, 0.6667, 0.4902))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 1, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7, 0.7, 0.7))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 2, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9, 0.9, 0.9))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 3, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0, 0))
            
            self.envs.append(env_handle)
            self.actor_handles_g1.append(actor_handle_g1)
            self.actor_handles_knob.append(actor_handle_knob)
        
        # self.print_g1_dof_index(self.gym, self.envs[0], self.actor_handles_g1[0])
        
        
    def _acquire_g1_asset_and_setup(self):
        
        # Acquire G1 asset
        asset_path = self.cfg.asset_g1.file.format(SDL_GYM_ROOT_DIR=SDL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options_g1 = gymapi.AssetOptions()
        asset_options_g1.default_dof_drive_mode = self.cfg.asset_g1.default_dof_drive_mode
        asset_options_g1.collapse_fixed_joints = self.cfg.asset_g1.collapse_fixed_joints
        asset_options_g1.replace_cylinder_with_capsule = self.cfg.asset_g1.replace_cylinder_with_capsule
        asset_options_g1.flip_visual_attachments = self.cfg.asset_g1.flip_visual_attachments
        asset_options_g1.fix_base_link = self.cfg.asset_g1.fix_base_link
        asset_options_g1.density = self.cfg.asset_g1.density
        asset_options_g1.angular_damping = self.cfg.asset_g1.angular_damping
        asset_options_g1.linear_damping = self.cfg.asset_g1.linear_damping
        asset_options_g1.max_angular_velocity = self.cfg.asset_g1.max_angular_velocity
        asset_options_g1.max_linear_velocity = self.cfg.asset_g1.max_linear_velocity
        asset_options_g1.armature = self.cfg.asset_g1.armature
        asset_options_g1.thickness = self.cfg.asset_g1.thickness
        asset_options_g1.disable_gravity = self.cfg.asset_g1.disable_gravity
        asset_options_g1.use_mesh_materials = self.cfg.asset_g1.use_mesh_materials
        
        g1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options_g1)
        self.g1_asset = g1_asset
        
        # G1 asset Setup
        self.num_dof_g1 = self.gym.get_asset_dof_count(g1_asset) # CONFUSE......
        self.num_bodies_g1 = self.gym.get_asset_rigid_body_count(g1_asset) # CONFUSE......
        self.dof_props_asset_g1 = self.gym.get_asset_dof_properties(g1_asset)
        self.rigid_shape_props_asset_g1 = self.gym.get_asset_rigid_shape_properties(g1_asset)
        
        for p in self.rigid_shape_props_asset_g1:
            p.friction = self.cfg.asset_knob.friction
        self.gym.set_asset_rigid_shape_properties(self.g1_asset, self.rigid_shape_props_asset_g1)
        
        # save body names from the g1 asset
        self.body_names_g1 = self.gym.get_asset_rigid_body_names(g1_asset)
        self.dof_names_g1 = self.gym.get_asset_dof_names(g1_asset)
        self.num_bodies_g1 = len(self.body_names_g1) # CONFUSE......
        self.num_dofs_g1 = len(self.dof_names_g1) # CONFUSE......
        ############################# Not Enabled Now #############################
        penalized_contact_names = []
        for name in self.cfg.asset_g1.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names_g1 if name in s])
        termination_contact_names = []
        for name in self.cfg.asset_g1.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names_g1 if name in s])
        ###########################################################################
        
        base_init_state_list_g1 = self.cfg.init_state_g1.pos + self.cfg.init_state_g1.rot + self.cfg.init_state_g1.lin_vel + self.cfg.init_state_g1.ang_vel
        self.base_init_state_g1 = to_torch(base_init_state_list_g1, device=self.device, requires_grad=False)
        self.start_pose_g1 = gymapi.Transform()
        self.start_pose_g1.p = gymapi.Vec3(*self.base_init_state_g1[:3])
        self.start_pose_g1.r = gymapi.Quat(*self.base_init_state_g1[3:7])
        
        self.actor_handles_g1 = []
    
    def _process_dof_props_g1(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs_g1, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs_g1, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs_g1, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
        
    def _acquire_knob_asset_and_setup(self):
        
        # Acquire Knob asset
        asset_path = self.cfg.asset_knob.file.format(SDL_GYM_ROOT_DIR=SDL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options_knob = gymapi.AssetOptions()
        asset_options_knob.default_dof_drive_mode = self.cfg.asset_knob.default_dof_drive_mode
        asset_options_knob.collapse_fixed_joints = self.cfg.asset_knob.collapse_fixed_joints
        asset_options_knob.replace_cylinder_with_capsule = self.cfg.asset_knob.replace_cylinder_with_capsule
        asset_options_knob.flip_visual_attachments = self.cfg.asset_knob.flip_visual_attachments
        asset_options_knob.fix_base_link = self.cfg.asset_knob.fix_base_link
        asset_options_knob.armature = self.cfg.asset_knob.armature
        asset_options_knob.disable_gravity = self.cfg.asset_knob.disable_gravity
        
        knob_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options_knob)
        self.knob_asset = knob_asset
        
        # Knob asset Setup
        self.num_dof_knob = self.gym.get_asset_dof_count(knob_asset) # CONFUSE......
        self.num_bodies_knob = self.gym.get_asset_rigid_body_count(knob_asset) # CONFUSE......
        self.dof_props_asset_knob = self.gym.get_asset_dof_properties(knob_asset)
        self.rigid_shape_props_asset_knob = self.gym.get_asset_rigid_shape_properties(knob_asset)
        
        for p in self.rigid_shape_props_asset_knob:
            p.friction = self.cfg.asset_g1.friction
        self.gym.set_asset_rigid_shape_properties(self.knob_asset, self.rigid_shape_props_asset_knob)
        
        # save body names from the knob asset
        self.body_names_knob = self.gym.get_asset_rigid_body_names(knob_asset)
        self.dof_names_knob = self.gym.get_asset_dof_names(knob_asset)
        self.num_bodies_knob = len(self.body_names_knob) # CONFUSE......
        self.num_dofs_knob = len(self.dof_names_knob) # CONFUSE......
        
        knnb_base_init_state_list = self.cfg.init_state_knob.pos + self.cfg.init_state_knob.rot + self.cfg.init_state_knob.lin_vel + self.cfg.init_state_knob.ang_vel
        self.base_init_state_knob = to_torch(knnb_base_init_state_list, device=self.device, requires_grad=False)
        self.start_pose_knob = gymapi.Transform()
        self.start_pose_knob.p = gymapi.Vec3(*self.base_init_state_knob[:3])
        
        self.actor_handles_knob = []   
        
    # def log_in_wandb(self, env_ids):
    #     if self.cfg.env.if_log_wandb and not self.cfg.env.test:
    #         self.total_reward_buf += self.rew_buf
            
    #         if len(env_ids) == 0:
    #             return
            
    #         wandb.log({
    #             "average_total_reward": self.total_reward_buf[env_ids].mean().item() / self.max_episode_length_s,
    #             **{f"average_{key}_reward": value for key, value in self.extras["episode"].items()}
    #         })
            
            # avg_total_reward = self.total_reward_buf[env_ids].mean().item()
            # avg_knob_rotate_reward = self.knob_rotate_reward_buf[env_ids].mean().item()
            # avg_contact_forces_reward = self.contact_forces_reward_buf[env_ids].mean().item()
            # avg_fingertips_TO_knob_center_dist_reward = self.fingertips_TO_knob_center_dist_reward_buf[env_ids].mean().item()
            
            # wandb.log({
            #     "average_total_reward": avg_total_reward,
            #     "average_knob_rotate_reward": avg_knob_rotate_reward,
            #     "average_contact_forces_reward": avg_contact_forces_reward,
            #     "average_fingertips_TO_knob_center_dist_reward": avg_fingertips_TO_knob_center_dist_reward
            # })
            
            # self.total_reward_buf[env_ids] = 0.
            # self.knob_rotate_reward_buf[env_ids] = 0.
            # self.contact_forces_reward_buf[env_ids] = 0.
            # self.fingertips_TO_knob_center_dist_reward_buf[env_ids] = 0.
            
    def log_in_wandb(self, env_ids):
        if self.cfg.env.if_log_wandb and not self.cfg.env.test:
            self.total_reward_buf += self.rew_buf

            if len(env_ids) == 0:
                return

            avg_total_reward = self.total_reward_buf[env_ids].mean().item() / self.max_episode_length_s
            sub_reward_sum = sum(value for value in self.extras["episode"].values())
            reward_sum_difference = avg_total_reward - sub_reward_sum

            wandb.log({
                "average_total_reward": avg_total_reward,
                **{f"average_{key}_reward": value for key, value in self.extras["episode"].items()},
                "reward_sum_difference": reward_sum_difference,  
            })
            
            self.total_reward_buf[env_ids] = 0.
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset DOF states
        """ Resets DOF position and velocities of selected environmments"""
        # self.dof_pos_g1[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof_g1), device=self.device)
        self.dof_pos_g1[env_ids] = self.default_dof_pos
        self.dof_vel_g1[env_ids] = 0.0
        
        # Reset knob to clockwise extreme
        self.dof_pos_knob[env_ids] = self.knob_initial_angle 
        self.dof_vel_knob[env_ids] = 0.0 
        
        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
            
        # reset buffers
        self.network_output_actions[env_ids] = 0.
        # self.real_dof_control_pos_target[env_ids] = 0.
        # self.last_network_output_actions[env_ids] = 0.
        # self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0 ################################################## CONFUSE...... 0? ##################################################
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        
        
        
        
        
        
        
        
        
        
        
    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
    def pre_physics_step(self, actions):
        network_output_actions = actions.clone().to(self.device) # shape = (num_envs, num_actions); values = [-1, 1]
        
        self.real_dof_control_pos_target[:, self.right_arm_dofs_num:] = self.dof_pos_g1[:, self.right_hand_index_0_joint_handle: self.right_hand_thumb_2_joint_handle+1] + network_output_actions[:, 6:] * self.dt * self.cfg.control.action_scale
        
        self._apply_actions_as_ctrl_targets(actions=network_output_actions[:, :6], do_scale=True)

        self.real_dof_control_pos_target[:, :] = tensor_clamp(
             self.real_dof_control_pos_target, self.dof_pos_limits[self.right_shoulder_pitch_joint_handle:, 0], self.dof_pos_limits[self.right_shoulder_pitch_joint_handle:, 1]
         )
        
    def _apply_actions_as_ctrl_targets(self, actions, do_scale=True, clamp_rot=True, clamp_rot_thresh=1.0e-6, do_force_ctrl=False):
        """Apply actions from policy as position/rotation targets."""
        
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor([0.05, 0.05, 0.05], device=self.device))
        self.ctrl_target_hand_pos = self.g1_right_hand_pos + pos_actions
        
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor([0.05, 0.05, 0.05], device=self.device))
        
        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = quat_from_angle_axis(angle, axis)
        if clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_hand_quat = quat_mul(rot_actions_quat, self.g1_right_hand_quat)
        # ############ Test ############
        # self.ctrl_target_hand_pos = self.global_knob_hand_reach_pos
        # self.ctrl_target_hand_quat = self.global_knob_hand_reach_rot
        # ############ Test ############
                
        # According Factory "if do_force_ctrl" part not included here
        # if do_force_ctrl:
        
        self.generate_ctrl_signals()
        
    def generate_ctrl_signals(self, jacobian_type='geometric', motor_ctrl_mode='gym'):
        """Get Jacobian. Set G1 DOF position targets or DOF torques."""
        
        # Get desired Jacobian
        if jacobian_type == 'geometric':
            self.hand_jacobian_tf = self.right_hand_jacobian
        elif jacobian_type == 'analytic':
            pass
        
        # Set PD joint pos target or joint torque
        if motor_ctrl_mode == 'gym':
            self._set_dof_pos_target()
        elif motor_ctrl_mode == 'manual':
            self._set_dof_torque()
            
    
    def _set_dof_pos_target(self):
         """Set G1 DOF position target to move hand towards target pose."""
         
         self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            num_envs=self.num_envs,
            arm_dof_pos=self.dof_pos_g1[:, self.right_shoulder_pitch_joint_handle: self.right_wrist_yaw_joint_handle+1],
            hand_pos=self.g1_right_hand_pos,
            hand_quat=self.g1_right_hand_quat,
            jacobian=self.hand_jacobian_tf,
            ctrl_target_hand_pos=self.ctrl_target_hand_pos,
            ctrl_target_hand_quat=self.ctrl_target_hand_quat,
            num_of_ctrl_target_dof_pos = 7,
            device=self.device)
         
         self.real_dof_control_pos_target[:, :self.right_arm_dofs_num] = self.ctrl_target_dof_pos
         
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        clip_actions = self.cfg.normalization.clip_actions
        self.network_output_actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.pre_physics_step(self.network_output_actions)
        
        # For development
        # self.visualize()

        # step physics and render each frame
        self.render()    
        
        for _ in range(self.cfg.control.decimation):
            
            # self._compute_torques(self.real_dof_control_pos_target)
            self._compute_dof_tartget_pos(self.real_dof_control_pos_target)
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.input_sim_actions_vectors))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
                
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def visualize(self):
        # Visualize the hand reach point
        self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # Hand link target pose
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], self.global_knob_hand_reach_pose)
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], self.global_knob_hand_reach_pose)
            
            # Knob center keypoint pose
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], self.global_knob_center_keypoint_pose)
            
            # Hand exact pose
            self.global_hand_pose.p = gymapi.Vec3(*self.g1_right_hand_pos[i].cpu().numpy())
            self.global_hand_pose.r = gymapi.Quat(*self.g1_right_hand_quat[i].cpu().numpy())
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], self.global_hand_pose)
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], self.global_hand_pose)
            
            
    ############################# Test #############################
    def _compute_dof_tartget_pos(self, actions):
        self.input_sim_actions_vectors[:, :self.right_shoulder_pitch_joint_handle] = 0
        self.input_sim_actions_vectors[:, self.right_shoulder_pitch_joint_handle:self.num_dof_g1] = actions
        
        # Just for test when handmade the start DoF pos
        # self.input_sim_actions_vectors[:, : self.num_dof_g1] = self.default_dof_pos
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.input_sim_actions_vectors))
    ############################# Test #############################

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # self.base_pos[:] = self.root_states_g1[:, 0:3]
        # self.base_quat[:] = self.root_states_g1[:, 3:7]
        # self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_g1[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_g1[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self._refresh_tensors()
        self.compute_all_useful_data_after_refresh()
        # self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) 
        self.log_in_wandb(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_reward()
        self.plotJuggler_plot()

        # self.last_network_output_actions[:] = self.network_output_actions[:]
        # self.last_dof_vel[:] = self.dof_vel_g1[:]
        # self.last_root_vel[:] = self.root_states_g1[:, 7:13]
        

    def compute_all_useful_data_after_refresh(self):
        # knob pose
        self.knob_pos = self.rigid_body_states[:, self.knob_handle][:, 0:3]
        self.knob_rot = self.rigid_body_states[:, self.knob_handle][:, 3:7] 
        
        # Compute knob hand reach pose
        self.hand_to_global_knob_hand_reach_pos = self.global_knob_hand_reach_pos - self.g1_right_hand_pos
        self.hand_to_global_knob_hand_reach_dist = torch.norm(self.hand_to_global_knob_hand_reach_pos, dim=-1)
        
        # compute knob angle and velocity
        self.knob_angle = self.dof_pos_knob  # current knob angle
        self.knob_angle_to_target = self.knob_target_angle - self.knob_angle  # difference between knob angle and target angle
        
        # Compute net force sensing reading
        self.right_hand_thumb_2_force_sensor_1_net_force = torch.norm(self.contact_forces[:, self.right_hand_thumb_2_force_sensor_1_link_handle, :], p=2, dim=-1)
        self.right_hand_thumb_2_force_sensor_2_net_force = torch.norm(self.contact_forces[:, self.right_hand_thumb_2_force_sensor_2_link_handle, :], p=2, dim=-1)
        self.right_hand_thumb_2_force_sensor_3_net_force = torch.norm(self.contact_forces[:, self.right_hand_thumb_2_force_sensor_3_link_handle, :], p=2, dim=-1)
        self.right_hand_index_1_force_sensor_1_net_force = torch.norm(self.contact_forces[:, self.right_hand_index_1_force_sensor_1_link_handle, :], p=2, dim=-1)
        self.right_hand_index_1_force_sensor_2_net_force = torch.norm(self.contact_forces[:, self.right_hand_index_1_force_sensor_2_link_handle, :], p=2, dim=-1)
        self.right_hand_index_1_force_sensor_3_net_force = torch.norm(self.contact_forces[:, self.right_hand_index_1_force_sensor_3_link_handle, :], p=2, dim=-1)
        
        self.thumb_force_sensor_3_TO_knob_center_keypoint_pos = self.global_knob_center_keypoint_pos - self.right_hand_thumb_2_force_sensor_3_pos
        self.thumb_force_sensor_3_TO_knob_center_keypoint_dist = torch.norm(self.thumb_force_sensor_3_TO_knob_center_keypoint_pos, dim=-1)
        self.index_force_sensor_3_TO_knob_center_keypoint_pos = self.global_knob_center_keypoint_pos - self.right_hand_index_1_force_sensor_3_pos
        self.index_force_sensor_3_TO_knob_center_keypoint_dist = torch.norm(self.index_force_sensor_3_TO_knob_center_keypoint_pos, dim=-1)
        # print(f'{self.thumb_force_sensor_3_TO_knob_center_keypoint_dist[0]}, {self.index_force_sensor_3_TO_knob_center_keypoint_dist[0]}')
        
        
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf
                
        self.reset_buf = (torch.squeeze(self.dof_pos_knob, -1) >= self.knob_target_angle)
        self.reset_buf |= (self.hand_to_global_knob_hand_reach_dist > 0.10)
        self.reset_buf |= self.time_out_buf # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(self.thumb_force_sensor_3_TO_knob_center_keypoint_dist > 0.05, self.index_force_sensor_3_TO_knob_center_keypoint_dist > 0.05)
        
    def plotJuggler_plot(self):
        self.data_publisher.publish({
            'current_knob_angle': self.dof_pos_knob[0],
            'right_hand_index_1_force_sensor_3_net_force': self.right_hand_index_1_force_sensor_3_net_force[0],
            'right_hand_thumb_2_force_sensor_3_net_force': self.right_hand_thumb_2_force_sensor_3_net_force[0],
        })
        
    def compute_observations(self):
        """ Computes observations
        """
        # sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        # cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        # self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             (self.dof_pos_g1 - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel_g1 * self.obs_scales.dof_vel,
        #                             self.network_output_actions,
        #                             # sin_phase,
        #                             # cos_phase
        #                             ),dim=-1)
        # self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             (self.dof_pos_g1 - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel_g1 * self.obs_scales.dof_vel,
        #                             self.network_output_actions,
        #                             # sin_phase,
        #                             # cos_phase
        #                             ),dim=-1)
        
        self.obs_buf = torch.cat((  
                                                self.g1_right_hand_pos, # Dim = 3
                                                get_euler_xyz_in_tensor(self.g1_right_hand_quat), # Dim = 3
                                                (self.g1_right_hand_7_dofs_pos - self.default_dof_pos[self.right_hand_index_0_joint_handle:self.right_hand_thumb_2_joint_handle+1]) * self.obs_scales.dof_pos, # Dim = 7
                                                self.network_output_actions, # Dim = 13
                                                self.hand_to_global_knob_hand_reach_dist.unsqueeze(-1), # Dim = 1
                                                self.knob_angle, # Dim = 1
                                                self.knob_angle_to_target, # Dim = 1
                                                self.right_hand_thumb_2_force_sensor_1_net_force.unsqueeze(-1), # Dim = 1
                                                self.right_hand_index_1_force_sensor_1_net_force.unsqueeze(-1), # Dim = 1
                                                                                                
                                                # sin_phase,
                                                # cos_phase
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((
                                                self.g1_right_hand_pos, # Dim = 3
                                                get_euler_xyz_in_tensor(self.g1_right_hand_quat), # Dim = 3
                                                (self.g1_right_hand_7_dofs_pos - self.default_dof_pos[self.right_hand_index_0_joint_handle:self.right_hand_thumb_2_joint_handle+1]) * self.obs_scales.dof_pos, # Dim = 7
                                                self.g1_right_hand_7_dofs_vel * self.obs_scales.dof_vel, # Dim = 7
                                                self.network_output_actions, # Dim = 13
                                                self.hand_to_global_knob_hand_reach_dist.unsqueeze(-1), # Dim = 1
                                                self.knob_angle, # Dim = 1
                                                self.knob_angle_to_target, # Dim = 1
                                                self.right_hand_thumb_2_force_sensor_1_net_force.unsqueeze(-1), # Dim = 1
                                                self.right_hand_index_1_force_sensor_1_net_force.unsqueeze(-1), # Dim = 1
                                                self.index_force_sensor_3_TO_knob_center_keypoint_dist.unsqueeze(-1), # Dim = 1
                                                self.thumb_force_sensor_3_TO_knob_center_keypoint_dist.unsqueeze(-1), # Dim = 1
                                                
                                                # sin_phase,
                                                # cos_phase
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # add noise if needed
        
        
    def _post_physics_step_callback(self):

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)        

    
    # def _compute_torques(self, actions):
    #     """ Compute torques from actions.
    #         Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
    #         [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

    #     Args:
    #         actions (torch.Tensor): Actions

    #     Returns:
    #         [torch.Tensor]: Torques sent to the simulation
    #     """
    #     #pd controller
    #     self.input_sim_actions_vectors[:, :self.right_shoulder_pitch_joint_handle] = 0
    #     actions_scaled = actions * self.cfg.control.action_scale
    #     control_type = self.cfg.control.control_type
    #     if control_type=="P":
    #         torques = self.p_gains[self.right_shoulder_pitch_joint_handle:self.num_dof_g1]*(actions_scaled + self.default_dof_pos.squeeze()[self.right_shoulder_pitch_joint_handle:self.num_dof_g1] - self.dof_pos_g1[:, self.right_shoulder_pitch_joint_handle:self.num_dof_g1]) - self.d_gains[self.right_shoulder_pitch_joint_handle:self.num_dof_g1]*self.dof_vel_g1[:, self.right_shoulder_pitch_joint_handle:self.num_dof_g1]
    #     elif control_type=="V":
    #         torques = self.p_gains[self.right_shoulder_pitch_joint_handle:self.num_dof_g1]*(actions_scaled - self.dof_vel_g1[self.right_shoulder_pitch_joint_handle:self.num_dof_g1]) - self.d_gains[self.right_shoulder_pitch_joint_handle:self.num_dof_g1]*(self.dof_vel_g1[self.right_shoulder_pitch_joint_handle:self.num_dof_g1] - self.last_dof_vel[self.right_shoulder_pitch_joint_handle:self.num_dof_g1])/self.sim_params.dt
    #     elif control_type=="T":
    #         torques = actions_scaled
    #     else:
    #         raise NameError(f"Unknown controller type: {control_type}")
        
    #     clipped_torques = torch.clip(torques, -self.torque_limits[self.right_shoulder_pitch_joint_handle:self.num_dof_g1], self.torque_limits[self.right_shoulder_pitch_joint_handle:self.num_dof_g1])
    #     self.input_sim_actions_vectors[:, self.right_shoulder_pitch_joint_handle:self.num_dof_g1] = clipped_torques
    
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew








    
    
        #------------ reward functions----------------
        
    def _reward_knob_rotation(self):
        this_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        dof_pos_knob = torch.squeeze(self.dof_pos_knob, -1)
        
        knob_rotation_percentage = torch.clamp(((dof_pos_knob - self.knob_initial_angle) / (self.knob_target_angle - self.knob_initial_angle)), 0.0, 1.0)
        this_reward += 10 * (knob_rotation_percentage - self.last_knob_rotation_percentage)
        self.last_knob_rotation_percentage = knob_rotation_percentage
        
        # # For WandB logging
        # if self.cfg.rewards.only_positive_rewards:
        #     self.knob_rotate_reward_buf += torch.clip(this_reward[:], min=0.)
        # else:
        #     self.knob_rotate_reward_buf += this_reward
        
        return this_reward

    def _reward_contact_forces(self):
        this_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        this_reward += torch.where(self.right_hand_index_1_force_sensor_3_net_force > 0.1, 0.0001, 0.0)
        this_reward += torch.where(self.right_hand_thumb_2_force_sensor_3_net_force > 0.1, 0.0001, 0.0)
        
        # # For WandB logging
        # if self.cfg.rewards.only_positive_rewards:
        #     self.contact_forces_reward_buf += torch.clip(this_reward[:], min=0.)
        # else:
        #     self.contact_forces_reward_buf += this_reward
        
        return this_reward
    
    def _reward_fingertips_TO_knob_center_dist(self):
        this_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        this_reward += 5 * (self.last_thumb_force_sensor_3_TO_knob_center_keypoint_dist - self.thumb_force_sensor_3_TO_knob_center_keypoint_dist)
        self.last_thumb_force_sensor_3_TO_knob_center_keypoint_dist = self.thumb_force_sensor_3_TO_knob_center_keypoint_dist
        
        this_reward += 5 * (self.last_index_force_sensor_3_TO_knob_center_keypoint_dist - self.index_force_sensor_3_TO_knob_center_keypoint_dist)
        self.last_index_force_sensor_3_TO_knob_center_keypoint_dist = self.index_force_sensor_3_TO_knob_center_keypoint_dist
                
        # # For WandB logging
        # if self.cfg.rewards.only_positive_rewards:
        #     self.fingertips_TO_knob_center_dist_reward_buf += torch.clip(this_reward[:], min=0.)
        # else:
        #     self.fingertips_TO_knob_center_dist_reward_buf += this_reward
        
        return this_reward
        
    def _reward_goal_reached_sparse(self):
        this_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        this_reward += torch.where(torch.squeeze(self.dof_pos_knob, -1) >= self.knob_target_angle, 1000, 0)
        
        return this_reward
    


    # # Inherit from G1 locomotion's reward functions, can use or not use them
    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.last_dof_vel - self.dof_vel_g1) / self.dt), dim=1)

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel_g1), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_network_output_actions - self.network_output_actions), dim=1)
    
    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.dof_pos_g1 - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    #     out_of_limits += (self.dof_pos_g1 - self.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)
   
    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.reset_buf * ~self.time_out_buf





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # Archived unused G1 locomotion reward functions
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)    
    
    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.base_lin_vel[:, 2])
    
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)    
    
    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self.root_states_g1[:, 2]
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)    
    
    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel_g1) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)    
    
    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)    
    
    # def _reward_alive(self):
    #     # Reward for staying alive
    #     return 1.0
    
    # def _reward_hip_pos(self):
    #     return torch.sum(torch.square(self.dof_pos_g1[:,[1,2,7,8]]), dim=1)