import torch

import envs.smp_env as smp_env
import engines.engine as engine
import util.torch_util as torch_util


class TaskVelocityEnv(smp_env.SMPEnv):
    """Tracks a body-frame velocity command [v_x, v_y] and a yaw rate command."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._tar_vx_min = env_config["tar_vx_min"]
        self._tar_vx_max = env_config["tar_vx_max"]
        self._tar_vy_min = env_config["tar_vy_min"]
        self._tar_vy_max = env_config["tar_vy_max"]
        self._tar_yaw_rate_min = env_config["tar_yaw_rate_min"]
        self._tar_yaw_rate_max = env_config["tar_yaw_rate_max"]
        self._tar_change_time_min = env_config["tar_change_time_min"]
        self._tar_change_time_max = env_config["tar_change_time_max"]
        # commands with planar speed below this (m/s) are resampled; 0 disables
        self._tar_speed_min = env_config.get("tar_speed_min", 0.0)

        self._reward_cmd_vel_w = float(env_config["reward_cmd_vel_w"])
        self._reward_cmd_yaw_w = float(env_config["reward_cmd_yaw_w"])
        self._reward_cmd_vel_scale = float(env_config["reward_cmd_vel_scale"])
        self._reward_cmd_yaw_scale = float(env_config["reward_cmd_yaw_scale"])

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._tar_marker_ids = []

        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        if (self._visualize):
            tar_marker_id = self._build_marker(env_id)

            if (env_id == 0):
                self._tar_marker_ids.append(tar_marker_id)
            else:
                assert(self._tar_marker_ids[0] == tar_marker_id)
        return

    def _build_marker(self, env_id):
        asset_file = "data/assets/objects/steering_marker.xml"
        tar_marker_id = self._engine.create_obj(env_id=env_id,
                                                obj_type=engine.ObjType.rigid,
                                                asset_file=asset_file,
                                                name="tar_marker",
                                                is_visual=True,
                                                enable_self_collisions=False,
                                                fix_root=True,
                                                color=[0.8, 0.0, 0.0])
        return tar_marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        self._tar_vel = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._tar_yaw_rate = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._prev_root_heading = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        return

    def _get_tar_marker_id(self):
        return self._tar_marker_ids[0]

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_char_root()
        return

    def _record_char_root(self):
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        self._prev_root_heading[:] = torch_util.calc_heading(self._engine.get_root_rot(char_id))
        return

    def _update_marker(self, env_ids):
        char_id = self._get_char_id()
        tar_marker_id = self._get_tar_marker_id()

        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        root_rot = self._engine.get_root_rot(char_id)[env_ids]
        tar_vel = self._tar_vel[env_ids]

        # the marker points along the commanded velocity, at a distance that grows with speed
        tar_vel3d = torch.cat([tar_vel, torch.zeros_like(tar_vel[..., 0:1])], dim=-1)
        heading_rot = torch_util.calc_heading_quat(root_rot)
        world_tar_vel = torch_util.quat_rotate(heading_rot, tar_vel3d)[..., 0:2]
        tar_speed = torch.norm(world_tar_vel, dim=-1, keepdim=True)
        tar_dir = world_tar_vel / tar_speed.clamp(min=1e-6)
        tar_dist = 0.5 + tar_speed

        marker_pos = root_pos.clone()
        marker_pos[..., 0:2] += tar_dist * tar_dir
        marker_pos[..., 2] = 0.0

        tar_theta = torch.atan2(tar_dir[..., 1], tar_dir[..., 0])
        axis = torch.zeros_like(root_pos)
        axis[..., -1] = 1.0
        marker_rot = torch_util.axis_angle_to_quat(axis, tar_theta)

        self._engine.set_root_pos(env_ids, tar_marker_id, marker_pos)
        self._engine.set_root_rot(env_ids, tar_marker_id, marker_rot)
        self._engine.set_root_vel(env_ids, tar_marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, tar_marker_id, 0.0)
        return

    def _update_task(self):
        reset_task_mask = self._time_buf >= self._tar_change_times
        reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        if len(reset_env_ids) > 0:
            self._reset_task(reset_env_ids)

        if (self._visualize):
            self._update_marker(self._env_ids)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        self._reset_tar(env_ids)

        n = env_ids.shape[0]
        rand_dt = torch.rand(n, device=self._device)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt

        if (self._visualize):
            self._update_marker(env_ids)
        return

    def _sample_tar_vel(self, n, max_rounds=100):
        vx = (self._tar_vx_max - self._tar_vx_min) * torch.rand(n, device=self._device) + self._tar_vx_min
        vy = (self._tar_vy_max - self._tar_vy_min) * torch.rand(n, device=self._device) + self._tar_vy_min
        if (self._tar_speed_min > 0):
            for _ in range(max_rounds):
                slow = torch.sqrt(vx * vx + vy * vy) < self._tar_speed_min
                num_slow = int(slow.sum().item())
                if (num_slow == 0):
                    break
                vx[slow] = (self._tar_vx_max - self._tar_vx_min) * torch.rand(num_slow, device=self._device) + self._tar_vx_min
                vy[slow] = (self._tar_vy_max - self._tar_vy_min) * torch.rand(num_slow, device=self._device) + self._tar_vy_min
        return vx, vy

    def _reset_tar(self, env_ids):
        n = env_ids.shape[0]

        vx, vy = self._sample_tar_vel(n)
        yaw_rate = (self._tar_yaw_rate_max - self._tar_yaw_rate_min) * torch.rand(n, device=self._device) + self._tar_yaw_rate_min

        self._tar_vel[env_ids] = torch.stack([vx, vy], dim=-1)
        self._tar_yaw_rate[env_ids] = yaw_rate
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        tar_vel = self._tar_vel
        tar_yaw_rate = self._tar_yaw_rate

        if (env_ids is not None):
            tar_vel = tar_vel[env_ids]
            tar_yaw_rate = tar_yaw_rate[env_ids]

        task_obs = torch.cat([tar_vel, tar_yaw_rate.unsqueeze(-1)], dim=-1)
        obs = torch.cat([obs, task_obs], dim=-1)
        return obs

    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)

        self._reward_buf[:] = compute_velocity_reward(root_pos=char_root_pos,
                                                      prev_root_pos=self._prev_root_pos,
                                                      root_rot=char_root_rot,
                                                      prev_root_heading=self._prev_root_heading,
                                                      tar_vel=self._tar_vel,
                                                      tar_yaw_rate=self._tar_yaw_rate,
                                                      dt=self._engine.get_timestep(),
                                                      vel_err_scale=self._reward_cmd_vel_scale,
                                                      yaw_err_scale=self._reward_cmd_yaw_scale,
                                                      vel_reward_w=self._reward_cmd_vel_w,
                                                      yaw_reward_w=self._reward_cmd_yaw_w)
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_velocity_reward(root_pos, prev_root_pos, root_rot, prev_root_heading, tar_vel, tar_yaw_rate, dt,
                            vel_err_scale, yaw_err_scale, vel_reward_w, yaw_reward_w):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float) -> Tensor

    root_vel = (root_pos - prev_root_pos) / dt
    heading_inv = torch_util.calc_heading_quat_inv(root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv, root_vel)[..., 0:2]

    vel_err = torch.sum(torch.square(tar_vel - local_root_vel), dim=-1)
    vel_reward = torch.exp(-vel_err_scale * vel_err)

    heading = torch_util.calc_heading(root_rot)
    yaw_rate = torch_util.normalize_angle(heading - prev_root_heading) / dt
    yaw_err = torch.square(tar_yaw_rate - yaw_rate)
    yaw_reward = torch.exp(-yaw_err_scale * yaw_err)

    reward = vel_reward_w * vel_reward + yaw_reward_w * yaw_reward
    return reward
