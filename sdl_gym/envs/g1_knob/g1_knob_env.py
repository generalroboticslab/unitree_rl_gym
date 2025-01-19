from sdl_gym import SDL_GYM_ROOT_DIR, envs, SDL_GYM_ENVS_DIR
from sdl_gym.envs.g1_knob.g1_robot import G1Robot
import os
import time

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1KnobRobot(G1Robot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
    
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
            dof_props_knob = self._process_dof_props_knob(self.dof_props_asset_knob, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle_knob, dof_props_knob)
            
            # Set knob colors
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8039, 0.6667, 0.4902))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 1, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7, 0.7, 0.7))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 2, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9, 0.9, 0.9))
            self.gym.set_rigid_body_color(env_handle, actor_handle_knob, 3, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0, 0))
            
            self.envs.append(env_handle)
            self.actor_handles_g1.append(actor_handle_g1)
            self.actor_handles_knob.append(actor_handle_knob)
            
        
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
        
        g1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options_g1)
        self.g1_asset = g1_asset
        
        # G1 asset Setup
        self.num_dof_g1 = self.gym.get_asset_dof_count(g1_asset) # CONFUSE......
        self.num_bodies_g1 = self.gym.get_asset_rigid_body_count(g1_asset) # CONFUSE......
        self.dof_props_asset_g1 = self.gym.get_asset_dof_properties(g1_asset)
        self.rigid_shape_props_asset_g1 = self.gym.get_asset_rigid_shape_properties(g1_asset)
        
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
        
    def _process_dof_props_knob(self, props, env_id):
        
        if env_id==0:
            self.knob_lower_limits = torch.tensor([-1.5707963267948966], device=self.device)  # Full clockwise rotation
        
        return props
        

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            # import ipdb; ipdb.set_trace()   
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    