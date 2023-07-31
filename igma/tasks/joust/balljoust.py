from typing import Tuple

import math
import numpy as np
import torch
from torch import Tensor
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import torch_rand_float
from igma.tasks.base.ma_vec_task import MultiAgentVecTask
from igma.functional import reset_any_team_all_terminated, reset_max_episode_length,\
    find_all_nearest_neighbors, select_by_env_index, is_same_team_index, rel_pos_by_env_index,\
    reward_agg_sum,\
    terminated_buf_update,\
    start_pos_uniform


class BallJoust(MultiAgentVecTask):

    def __init__(self, cfg, sim_device, **kwargs):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # num_obs = 10
        num_obs = 7
        n_agts = cfg["env"].get("numAgentsPerTeam", 1) * cfg["env"].get("numTeams", 1)
        num_rng = min(n_agts - 1, 3)
        num_obs += (num_obs + 1 + 3) * num_rng
        self.num_rng = num_rng

        # Actions:
        # num_acts = 4
        num_acts = 3

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.thrust_action_speed_scale = 400

        self.radius = 3.0 + n_agts
        self.center = torch.tensor(cfg["env"].get("centerPos", [0, 0, 1. + self.radius]), device=sim_device)

        super().__init__(config=self.cfg, sim_device=sim_device, **kwargs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()

        # max_thrust = 8
        # max_thrust = 4
        # max_thrust = 2
        # self.thrust_lower_limits = torch.zeros(1, device=self.device, dtype=torch.float32)
        # self.thrust_upper_limits = max_thrust * torch.ones(1, device=self.device, dtype=torch.float32)

        self.thrust_lower_limit = 0
        # self.thrust_upper_limit = 1000
        self.thrust_upper_limit = 8000
        self.thrust_lateral_component = 0.7

        # control tensors
        # self.dirs = torch.zeros((self.num_agts, 2), dtype=torch.float32, device=self.device, requires_grad=False)
        # self.thrusts = torch.zeros((self.num_agts, ), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_agts, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_agts, dtype=torch.int32, device=self.device)

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 2.5)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        # self.sim_params.gravity.z = -9.81
        self.sim_params.gravity.z = 0
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_balljoust_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_balljoust_asset(self):
        chassis_radius = 0.1

        root = ET.Element('mujoco')
        root.attrib["model"] = "Ballcopter"
        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        worldbody = ET.SubElement(root, "worldbody")

        chassis = ET.SubElement(worldbody, "body")
        chassis.attrib["name"] = "chassis"
        chassis.attrib["pos"] = "%g %g %g" % (0, 0, 0)
        chassis_geom = ET.SubElement(chassis, "geom")
        chassis_geom.attrib["type"] = "sphere"
        chassis_geom.attrib["size"] = "%g" % (chassis_radius)
        chassis_geom.attrib["pos"] = "0 0 0"
        chassis_geom.attrib["density"] = "50"
        chassis_joint = ET.SubElement(chassis, "joint")
        chassis_joint.attrib["name"] = "root_joint"
        chassis_joint.attrib["type"] = "free"

        gymutil._indent_xml(root)
        ET.ElementTree(root).write("ball.xml")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "."
        asset_file = "ball.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        # asset_options.linear_damping = 1.0
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # pos = start_pos_circle(
        #     self.num_envs, self.num_agents, gymapi.UP_AXIS_Z, radius_coef=0.2, randomize_coef=0
        # )
        pos = start_pos_uniform(self.num_envs, self.num_agents, radius=self.radius / 2.)
        pos += self.center.cpu()

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.add_env(env)

            for k in range(self.num_agents):
                pose = gymapi.Transform()
                pose.p += gymapi.Vec3(*pos[i][k].tolist())
                actor_handle = self.gym.create_actor(env, asset, pose, "balljoust", i, 0, 0)
                # pretty colors
                team = self.get_team_id(k)
                chassis_color = gymapi.Vec3(0.8, 0.6, 0.2) + self.team_colors[team]
                self.gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)
                self.add_actor(actor_handle)

        if self.debug_viz:
            # need env offsets for the rotors
            self.env_offsets = self.get_env_offsets().view(-1, 3)

    def reset_idx(self, env_ids):
        agt_ids = self.get_reset_agent_ids(env_ids)
        num_resets = len(agt_ids)

        actor_indices = self.all_actor_indices[agt_ids].flatten()

        self.root_states[agt_ids] = self.initial_root_states[agt_ids]
        self.root_states[agt_ids, 0] += torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        self.root_states[agt_ids, 1] += torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        self.root_states[agt_ids, 2] += torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_state_tensor,
                                                     gymtorch.unwrap_tensor(actor_indices), num_resets)
        # self.dirs[agt_ids] = torch_rand_float(-0.5, 0.5, (num_resets, 2), self.device)

        super().reset_idx(env_ids)

    def pre_physics_step(self, _actions):
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        agt_ids = self.get_reset_agent_ids(reset_env_ids)

        self.terminate_agents()

        actions = _actions.to(self.device)
        actions = actions.view(self.num_agts, -1)
        actions = self.clear_terminated(actions)

        # self.thrusts += self.dt * self.thrust_action_speed_scale * actions[:, -1]
        # self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)
        # self.thrusts = self.clear_terminated(self.thrusts)

        # force_shifts = self.dt * 8 * math.pi * actions[:, 0:3]
        # force_shifted = torch.nn.functional.normalize((self.forces + force_shifts), p=2.0, dim=-1)
        # self.forces = force_shifted * self.thrusts.unsqueeze(-1)

        # self.dirs = torch.clamp(self.dirs + self.dt * 8.0 * actions[:, 0:2], -1., 1.)
        # self.dirs = torch.clamp(actions[:, 0:2], -1., 1.)

        # azm = self.dirs[:, 0] * math.pi
        # elv = self.dirs[:, 1] * 0.5 * math.pi
        # self.forces[:, 0] = azm.cos() * elv.cos()
        # self.forces[:, 1] = azm.sin() * elv.cos()
        # self.forces[:, 2] = elv.sin()
        # self.forces *= self.thrusts.unsqueeze(-1)
        # print(self.dirs)
        # print(self.thrusts)
        # print(self.forces)

        # self.forces = torch.nn.functional.normalize(actions[:, 0:3], p=2.0, dim=-1) * self.thrusts.unsqueeze(-1)

        thrust_action_speed_scale = 2000
        tul = self.thrust_upper_limit
        # tlc = self.thrust_lateral_component

        # vertical_thrust_prop_0 = torch.clamp(actions[:, 2] * thrust_action_speed_scale, -tul, tul)
        # lateral_fraction_prop_0 = torch.clamp(actions[:, 0:2], -tlc, tlc)

        # v_prop = (1 - lateral_fraction_prop_0.norm(p=2, dim=-1))
        # self.forces[:, 2] = self.dt * vertical_thrust_prop_0 * v_prop
        # self.forces[:, 0:2] = self.forces[:, 2, None] * lateral_fraction_prop_0
        self.forces[:, 0:3] = self.dt * torch.clamp(actions[:, 0:3] * thrust_action_speed_scale, -tul, tul)

        # self.thrusts[:, 0, 2] = self.dt * vertical_thrust_prop_0
        # self.thrusts[:, 0, 0:2] = self.thrusts[:, 0, 2, None] * lateral_fraction_prop_0

        # self.forces[:, 1] = self.thrusts[:, 0]
        # self.forces[:, 3] = self.thrusts[:, 1]

        # clear actions for reset envs
        # self.thrusts[agt_ids] = 0.0
        self.forces[agt_ids] = 0.0

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        self.update_progress()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            starts = self.root_states[:, 0:3] + self.env_offsets
            # ends = starts + 0.1 * self.forces
            ends = self.env_offsets + self.center

            # submit debug line geometry
            verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
            colors = np.zeros((self.num_agts, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            # colors[self.terminated_buf] = [0, 1, 0]
            cod_buf = self.cod_buf.cpu()

            colors[(cod_buf == 1).numpy()] = [0, 1, 0]
            colors[(cod_buf == 3).numpy()] = [0, 0, 1]
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_agts, verts, colors)

    def compute_observations(self):
        self.obs_buf[:] = compute_balljoust_observations(
            self.obs_buf,
            self.root_states,
            # self.thrusts,
            # self.dirs,
            self.forces,
            self.terminated_buf,
            self.center,
            self.num_envs,
            self.num_agents,
            self.num_rng,
            self.num_teams,
        )
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.terminated_buf[:], self.cod_buf[:] = compute_balljoust_reward(
            self.root_states,
            self.reset_buf,
            self.terminated_buf,
            self.cod_buf,
            self.progress_buf,
            self.reward_weight,
            self.center,
            # self.thrusts,
            self.forces,
            self.max_episode_length,
            self.num_envs,
            self.num_agents,
            self.num_teams,
            self.value_size)


@torch.jit.script
def compute_balljoust_observations(
    obs_buf: Tensor,
    root_states: Tensor,
    # thrusts: Tensor,
    # dirs: Tensor,
    forces: Tensor,
    terminated_buf: Tensor,
    center: Tensor,
    num_envs: int,
    num_agents: int,
    num_rng: int,
    num_teams: int,
) -> Tensor:
    root_positions = root_states[..., 0:3]
    root_dist = torch.cdist(root_positions, center.unsqueeze(0)).squeeze()
    # root_quats = root_states[..., 3:7]
    root_linvels = root_states[..., 7:10]
    # root_angvels = root_states[..., 10:13]
    n_obs = 7
    obs_buf = torch.zeros_like(obs_buf)
    # obs_buf[..., 0:3] = root_positions
    obs_buf[..., 0:3] = (root_positions - center) / 3
    obs_buf[..., 3:6] = root_linvels / 2
    obs_buf[..., 6] = root_dist
    # obs_buf[..., 3] = root_dist
    # obs_buf[..., 4:6] = dirs
    # obs_buf[..., 6] = thrusts
    # obs_buf[..., 4:7] = forces
    # obs_buf[..., 3:7] = root_quats
    # obs_buf[..., 7:10] = root_linvels / 2
    # obs_buf[..., 10:13] = root_angvels / math.pi
    # obs_buf[..., 13] = root_dist
    # obs_buf[..., 14:16] = dirs

    if num_agents > 1:
        max_dist = 1.
        ind = find_all_nearest_neighbors(root_positions, num_envs, k=num_rng, max_dist=max_dist)
        rel_obs_obs = select_by_env_index(obs_buf[..., :n_obs], ind, num_envs, num_agents)
        rel_pos_obs = rel_pos_by_env_index(root_positions, ind, num_envs, num_agents)
        team_obs = is_same_team_index(ind, num_envs, num_agents, num_teams).float()
        ma_obs = torch.cat((
            team_obs.unsqueeze(-1),
            rel_pos_obs,
            rel_obs_obs,
        ), dim=-1)
        ma_obs[ind == -1] = 0
        obs_buf[..., n_obs:] = ma_obs.view(obs_buf.shape[0], -1)

    obs_buf[terminated_buf, :] = 0

    return obs_buf


@torch.jit.script
def compute_balljoust_reward(
    root_states: Tensor,
    reset_buf: Tensor,
    terminated_buf: Tensor,
    cod_buf: Tensor,
    progress_buf: Tensor,
    reward_weight: Tensor,
    center: Tensor,
    # thrusts: Tensor,
    forces: Tensor,
    max_episode_length: int,
    num_envs: int,
    num_agents: int,
    num_teams: int,
    value_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    root_positions = root_states[..., 0:3]
    # torso_vel = root_states[:, 7:10]
    z_pos = root_states[..., 2]
    # distance to target
    # root_dist = torch.cdist(root_positions, center.unsqueeze(0)).squeeze()
    root_dist = (root_positions - center).norm(p=2, dim=-1)
    # print(root_positions[0])
    # print(center)
    # pos_reward_coef = 0.1
    pos_reward_coef = 1.0
    pos_reward = pos_reward_coef / (1.0 + root_dist**2)
    # pos_reward = pos_reward_coef * torch.exp(-root_dist)
    # pos_reward = 0

    # uprightness
    # root_quats = root_states[..., 3:7]
    # ups = quat_axis(root_quats, 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    # up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    # root_angvels = root_states[..., 10:13]
    # spinnage = torch.abs(root_angvels[..., 2])
    # spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # alive_reward = torch.ones_like(z_pos) * 0.5
    # thrusts_cost = (-0.005) * thrusts ** 2
    thrusts_cost = (-0.005) * forces.norm(p=2, dim=-1)

    # move_center_reward = 0.1 * torch.sum(
    #     -(root_positions / (torch.norm(root_positions, dim=-1, keepdim=True) + 1e-7))
    #     * (torso_vel / (torch.norm(torso_vel, dim=-1, keepdim=True) + 1e-7)),
    #     dim=-1
    # )     # [E * N]

    # combined reward
    # uprigness and spinning only matter when close to the target
    # + z_pos * 0.5 \
    # + spinnage_reward * 0.5 \
    reward = 0 \
        + pos_reward \
        + thrusts_cost \
        # + move_center_reward \

    # + alive_reward \
    # + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    boundary = 3.0 + num_agents
    ones = torch.ones_like(z_pos, dtype=torch.bool)
    terminated = torch.zeros_like(z_pos, dtype=torch.bool)
    terminated = torch.where(root_dist > boundary, ones, terminated)
    cod_buf = torch.where(root_dist > boundary, ones * 1, cod_buf)
    # terminated = torch.where(root_positions[..., 2] < 0.3, ones, terminated)
    # cod_buf = torch.where(root_positions[..., 2] < 0.3, ones*2, cod_buf)

    # suicide_cost = -10.0
    suicide_cost = -1000.0
    reward = torch.where(terminated, torch.ones_like(reward) * suicide_cost, reward)

    if num_agents > 1:
        pos = root_positions.view(num_envs, num_agents, -1)
        dist = torch.cdist(pos, pos)  # [E, N, N]
        val, ind = [r[:, :, 1] for r in torch.topk(dist, 2, largest=False)]  # [E, N]
        env_idx = torch.arange(0, num_envs).view(num_envs, 1)
        target_alive = torch.logical_not(terminated_buf.view(num_envs, num_agents)[env_idx, ind])
        precond = torch.logical_and(target_alive, val < 0.25)
        # z_pos = z_pos.view(num_envs, num_agents)
        # tgt_z_pos = z_pos[env_idx, ind]
        # jousted = torch.logical_and(precond, tgt_z_pos - z_pos > 0.1)
        # jouster = torch.logical_and(precond, z_pos - tgt_z_pos > 0.1)
        root_dist = root_dist.view(num_envs, num_agents)
        tgt_dist = root_dist[env_idx, ind]
        jousted = torch.logical_and(precond, tgt_dist - root_dist > 0.1)
        jouster = torch.logical_and(precond, root_dist - tgt_dist > 0.1)
        terminated = torch.where(jousted.flatten(), ones, terminated)
        cod_buf = torch.where(jousted.flatten(), ones * 3, cod_buf)
        killed_cost = -2.0
        reward = torch.where(jousted.flatten(), torch.ones_like(reward) * killed_cost, reward)
        joust_score = 1000
        # joust_score = 0
        same_team = is_same_team_index(ind.unsqueeze(-1), num_envs, num_agents, num_teams).squeeze()
        ek_jouster = torch.logical_and(torch.logical_not(same_team), jouster)
        reward[ek_jouster.flatten()] += joust_score
        # tk_jouster = torch.logical_and(torch.logical_not(same_team), jouster)
        # reward[tk_jouster.flatten()] -= joust_score
        # reward[jousted] -= joust_score
        # rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        # move_opp_reward = 0.1 * torch.sum(
        #     (rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-7))
        #     * (torso_vel / (torch.norm(torso_vel, dim=-1, keepdim=True) + 1e-7)).unsqueeze(2),
        #     dim=[-1, -2]
        # )     # [E, N]

    reward = torch.where(terminated_buf, torch.zeros_like(reward), reward)

    reward = reward_agg_sum(reward, num_envs, value_size)

    # reward = reward_reweight_team(reward, reward_weight)

    terminated_buf = terminated_buf_update(terminated_buf, terminated)

    reset = reset_any_team_all_terminated(reset_buf, terminated_buf, num_envs, num_teams)
    # resets due to episode length
    # reset = torch.where(progress_buf >= max_episode_length - 1, ones, terminated)
    reset = reset_max_episode_length(reset, progress_buf, num_envs, max_episode_length)

    return reward, reset, terminated_buf, cod_buf
