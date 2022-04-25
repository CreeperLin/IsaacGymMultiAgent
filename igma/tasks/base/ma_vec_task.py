from typing import Dict, List, Any, Optional, Tuple
import sys
import math
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from isaacgymenvs.tasks.base.vec_task import VecTask
from torch import Tensor


@torch.jit.script
def obs_all_nearest_neighbors(
    pos: Tensor,
    num_envs: int,
    num_agents: int,
    max_num_obs: int,
    max_dist: float = 0.,
) -> Tensor:
    pos = pos.view(num_envs, num_agents, -1)   # [E, N, 3]
    dist = torch.cdist(pos, pos)    # [E, N, N]
    val, ind = torch.topk(dist, max_num_obs + 1, largest=False)     # [E, N, M+1]
    if max_dist:
        ind[val > max_dist] = -1
    ind = ind[:, :, 1:]     # [E, N, M]
    return ind


@torch.jit.script
def obs_get_by_env_index(
    obs: Tensor,
    ind: Tensor,
    num_envs: int,
    num_agents: int,
) -> Tensor:
    return obs.view(num_envs, num_agents, -1)[
        torch.arange(0, num_envs).view(num_envs, 1, 1), ind
    ]   # [E, N, M, S]


@torch.jit.script
def obs_rel_pos_by_env_index(
    pos: Tensor,
    ind: Tensor,
    num_envs: int,
    num_agents: int,
) -> Tensor:
    epos = pos.view(num_envs, num_agents, -1)   # [E, N, 3]
    rel_pos = epos.unsqueeze(2) - epos.unsqueeze(1)    # [E, N, N, 3]
    return rel_pos[
        torch.arange(0, num_envs).view(num_envs, 1, 1), torch.arange(0, num_agents).view(1, num_agents, 1), ind
    ]   # [E, N, M, 3]


@torch.jit.script
def obs_same_team_index(
    ind: Tensor,
    num_envs: int,
    num_agents: int,
    num_teams: int
) -> Tensor:
    num_agts_team = num_agents // num_teams
    return (
        (torch.arange(0, num_agents, device=ind.device) // num_agts_team).repeat(num_envs, 1).unsqueeze(-1)
        == (ind // num_agts_team)
    )   # [E, N, M]


@torch.jit.script
def reset_any_team_all_terminated(
    reset: Tensor,
    terminated: Tensor,
    num_envs: int,
    num_teams: int,
) -> Tensor:
    reset_ones = torch.ones_like(reset)
    return torch.where(
        torch.any(torch.all(terminated.view(num_envs, num_teams, -1), dim=-1), dim=-1), reset_ones, reset
    )


@torch.jit.script
def reset_max_episode_length(
    reset: Tensor,
    progress_buf: Tensor,
    num_envs: int,
    max_episode_length: int,
) -> Tensor:
    reset_ones = torch.ones_like(reset)
    return torch.where(
        torch.all(progress_buf.view(num_envs, -1) >= max_episode_length - 1, dim=-1), reset_ones, reset
    )


@torch.jit.script
def reward_reweight_team(
    reward: Tensor,
    reward_weight: Tensor,
) -> Tensor:
    value_size = reward_weight.shape[0]
    if value_size > 1:
        reward = (reward.view(-1, value_size) @ reward_weight).flatten()
    return reward


@torch.jit.script
def reward_agg_sum(
    reward: Tensor,
    num_envs: int,
    value_size: int,
) -> Tensor:
    return torch.sum(reward.view(num_envs, value_size, -1), dim=-1).flatten()


@torch.jit.script
def terminated_buf_update(
    terminated_buf: Tensor,
    terminated: Tensor,
) -> Tensor:
    return torch.logical_or(terminated_buf, terminated.flatten())


def start_pos_circle(
    num_envs,
    num_agents,
    up_axis_idx=gymapi.UP_AXIS_Z,
    radius_coef=0.75,
    randomize_coef=1.,
) -> Tensor:
    init_pos_radius = num_agents * radius_coef
    pos = torch.arange(0, num_agents) * 2. * math.pi / num_agents
    pos = pos.repeat(3, num_envs).T.view(num_envs, num_agents, -1)
    if up_axis_idx is gymapi.UP_AXIS_Z:
        x_axis, y_axis, up_axis = [0, 1, 2]
    else:
        raise NotImplementedError
    pos[:, :, x_axis].cos_()
    pos[:, :, y_axis].sin_()
    pos.mul_(init_pos_radius)
    if randomize_coef:
        pos += randomize_coef * torch.randn(num_envs, num_agents, 3)
    pos[:, :, up_axis] = 0
    return pos


def start_pos_normal(
    num_envs,
    num_agents,
    radius=1,
) -> Tensor:
    pos = torch.randn(num_envs, num_agents, 3) * radius
    return pos


def start_pos_uniform(
    num_envs,
    num_agents,
    radius=1,
) -> Tensor:
    pos = torch.rand(num_envs, num_agents, 3) * radius * 2 - radius
    return pos


class MultiAgentVecTask(VecTask):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.num_teams = config["env"].get("numTeams", 1)
        self.num_agents_team = config["env"].get("numAgentsPerTeam", 1)
        self.num_agents = self.num_agents_team * self.num_teams
        config["env"]["numAgents"] = self.num_agents
        self.reward_sum = config["env"].get("rewardSum", 'team')
        zero_sum = config["env"].get("rewardZeroSum", True)
        if self.reward_sum == 'team':
            value_size = self.num_teams
        elif self.reward_sum == 'all':
            value_size = 1
        elif self.reward_sum in [None, 'none']:
            value_size = self.num_agents
        self.value_size = value_size
        self.num_agents_export = value_size
        self.value_size_export = self.value_size // self.num_agents_export
        space_mult = self.num_agents // self.num_agents_export
        self.num_obs_per_agent = config["env"]["numObservations"]
        self.num_acts_per_agent = config["env"]["numActions"]
        config["env"]["numObservations"] = self.num_obs_per_agent * space_mult
        config["env"]["numActions"] = self.num_acts_per_agent * space_mult
        team_colors = [
            (0.97, 0.38, 0.06),
            (0.38, 0.06, 0.97),
            (0.06, 0.97, 0.38),
        ]
        self.team_colors = [gymapi.Vec3(*t) for t in team_colors]
        self.num_agts = config["env"]["numEnvs"] * self.num_agents

        self.viewer = None
        self.enable_viewer_sync = config["env"].get("rewardZeroSum", True)
        self.viewer_render_collision = False
        self.actor_handles = []
        self.env_handles = []
        self.callbacks = {}
        self.cam_pos = [20.0, 25.0, 10.0]
        self.cam_target = [10.0, 15.0, 0.0]

        super().__init__(
            config=config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless
        )

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor).view(self.num_agts, -1)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        if self.dof_state_tensor.ndim == 0:
            self.dof_states = None
        else:
            self.dof_states = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_agts, -1, 2)

        reward_weight = torch.ones((value_size, value_size), device=self.device, dtype=torch.float)
        if zero_sum and value_size > 1:
            reward_weight *= (-1 / (value_size-1))
            reward_weight[[torch.arange(0, value_size)] * 2] = 1.
        self.reward_weight = reward_weight
        self.terminated_buf = torch.zeros(self.num_envs * self.num_agents, device=self.device, dtype=torch.bool)
        self.cod_buf = torch.zeros(self.num_envs * self.num_agents, device=self.device, dtype=torch.long)

        save_replay_episodes = config["env"].get("saveReplayEpisodes", 0)
        self.save_replay_path = config["env"].get("saveReplayPath", "./replay.pt")
        self.save_replay = save_replay_episodes > 0
        self.save_replay_episodes = save_replay_episodes
        self.replay_mode = False
        if self.save_replay:
            self.replay_episode_pt = -1
            self.replay_action_pt = -1
            self.replay_device = 'cpu'
            self.replay_actions = torch.zeros(
                (save_replay_episodes, self.max_episode_length, self.num_agents_export, self.num_actions),
                device=self.replay_device, dtype=torch.float
            )
            self.replay_root_states = torch.zeros(
                (save_replay_episodes, self.num_agents, 13),
                device=self.replay_device, dtype=torch.float
            )
            if self.dof_states is not None:
                self.replay_dof_states = torch.zeros(
                    (save_replay_episodes, self.num_agents, self.dof_states.shape[1], 2),
                    device=self.replay_device, dtype=torch.float
                )
        self.exec_callback('init')

    def add_actor(self, actor_handle):
        self.actor_handles[-1].append(actor_handle)

    def add_env(self, env_handle):
        self.env_handles.append(env_handle)
        self.actor_handles.append([])

    def get_actor(self, env_id, actor_id):
        return self.actor_handles[env_id][actor_id]

    def get_env(self, env_id):
        return self.env_handles[env_id]

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            # (self.num_envs * self.num_agents_export, self.num_obs), device=self.device, dtype=torch.float)
            (self.num_envs * self.num_agents, self.num_obs_per_agent), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        if self.value_size_export == 1:
            rew_size = self.num_envs * self.num_agents_export
        else:
            rew_size = (self.num_envs * self.num_agents_export, self.value_size_export)
        self.rew_buf = torch.zeros(
            rew_size, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            # (self.num_envs), device=self.device, dtype=torch.long)
            (self.num_envs * self.num_agents_export), device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            # (self.num_envs), device=self.device, dtype=torch.long)
            (self.num_envs * self.num_agents_export), device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        # actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)
        actions = torch.zeros(
            (self.num_envs * self.num_agents_export, self.num_actions), device=self.rl_device, dtype=torch.float
        )

        return actions

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    def get_team_id(self, agt_id):
        return agt_id // self.num_agents_team

    def get_reset_agent_ids(self, env_ids):
        if self.num_agents == 1:
            return env_ids
        ent_ids = (env_ids * self.num_agents).repeat(self.num_agents).view(self.num_agents, -1)
        ent_ids += torch.arange(0, self.num_agents).view(-1, 1).to(device=self.device)
        return ent_ids.flatten().to(dtype=torch.long)

    def get_env_offsets(self):
        env_offsets = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device)
        for i in range(self.num_envs):
            env_origin = self.gym.get_env_origin(self.env_handles[i])
            env_offsets[i, :, :] = torch.tensor([env_origin.x, env_origin.y, env_origin.z])
        return env_offsets

    def add_callback(self, key, cb):
        cbs = self.callbacks.get(key, None)
        if cbs is None:
            cbs = []
            self.callbacks[key] = cbs
        cbs.append(cb)

    def exec_callback(self, key, *args, **kwargs):
        cbs = self.callbacks.get(key)
        if cbs is None:
            return
        for cb in cbs:
            cb(self, *args, **kwargs)

    def reset_idx(self, env_ids):
        self.exec_callback('reset_idx', env_ids)
        if self.replay_mode:
            self.replay_episode_pt += 1
            print('load', self.replay_episode_pt, self.replay_action_pt)
            self.root_states[:self.num_agents] = self.replay_root_states[self.replay_episode_pt].to(self.device)
            self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensor)
            if self.dof_states is not None:
                self.dof_states[:self.num_agents] = self.replay_dof_states[self.replay_episode_pt].to(self.device)
                self.gym.set_dof_state_tensor(self.sim, self.dof_state_tensor)
            self.replay_action_pt = -1
            if self.replay_episode_pt >= len(self.replay_root_states) - 1:
                self.replay_mode = False
        elif self.save_replay and 0 in env_ids:
            self.replay_episode_pt += 1
            print('save', self.replay_episode_pt, self.replay_action_pt)
            root_states = self.root_states[:self.num_agents].to(self.replay_device)
            self.replay_root_states[self.replay_episode_pt] = root_states
            if self.dof_states is not None:
                dof_states = self.dof_states[:self.num_agents].to(self.replay_device)
                self.replay_dof_states[self.replay_episode_pt] = dof_states
            self.replay_action_pt = -1
            if self.replay_episode_pt >= self.save_replay_episodes - 1:
                self.save_replay_state_dict(self.save_replay_path)
                self.save_replay = False
        self.progress_buf.view(self.num_envs, -1)[env_ids] = 0
        self.reset_buf.view(self.num_envs, -1)[env_ids] = 0
        self.terminated_buf.view(self.num_envs, -1)[env_ids] = 0
        self.cod_buf.view(self.num_envs, -1)[env_ids] = 0

    def get_obs_export(self, obs_buf):
        return torch.clamp(obs_buf, -self.clip_obs, self.clip_obs)\
            .view(-1, self.num_obs)\
            .to(self.rl_device)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.replay_mode:
            self.replay_action_pt += 1
            actions = self.replay_actions[self.replay_episode_pt][self.replay_action_pt]
        elif self.save_replay:
            self.replay_action_pt += 1
            replay_actions = actions[:self.num_agents_export].to(self.replay_device)
            self.replay_actions[self.replay_episode_pt][self.replay_action_pt] = replay_actions
        self.exec_callback('pre_step')
        obs_dict, rew_buf, reset_buf, extras = super().step(actions)
        self.exec_callback('post_step')
        if self.num_agents_export > 1:
            obs_dict['obs'] = obs_dict['obs'].view(self.num_envs * self.num_agents_export, self.num_obs)
            reset_buf = reset_buf.repeat(self.num_agents_export)
            # reset_buf = torch.all(self.terminated_buf.view(self.num_envs, self.num_agents_export, -1), dim=-1)
        return obs_dict, rew_buf, reset_buf, extras

    def update_progress(self):
        self.progress_buf[
            torch.logical_not(
                torch.all(self.terminated_buf.view(self.num_envs, self.num_agents_export, -1), dim=-1)
            ).flatten()
        ] += 1

    def select_terminated(self, x):
        return x.view(self.num_envs * self.num_agents, -1)[self.terminated_buf]

    def clear_terminated(self, x):
        x.view(self.num_envs * self.num_agents, -1)[self.terminated_buf] = 0
        return x

    def terminate_agents(self):
        term_list = torch.nonzero(self.terminated_buf.view(self.num_envs, self.num_agents), as_tuple=False).tolist()
        for env_id, actor_id in term_list:
            self.terminate_agent(env_id, actor_id)

    def terminate_agent(self, env_id, actor_id):
        pass

    def reset(self, indices: Optional[List[int]] = None) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        if indices is not None:
            self.reset_idx(indices)
            self.compute_observations()
            return {'obs': self.get_obs_export(self.obs_buf.view(self.num_envs, -1)[indices])}

        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = self.get_obs_export(self.obs_buf)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        self.exec_callback('reset')

        return self.obs_dict

    def set_viewer(self):
        """Create the viewer."""
        # if running with a viewer, set up keyboard shortcuts and camera
        cam_pos = self.cam_pos
        cam_target = self.cam_target
        if self.headless is False:
            cam_props = gymapi.CameraProperties()
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, cam_props)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)

            if sim_params.up_axis != gymapi.UP_AXIS_Z:
                cam_pos = [cam_pos[0], cam_pos[2], cam_pos[1]]
                cam_target = [cam_target[0], cam_target[2], cam_target[1]]

            env = self.get_env(0)
            self.gym.viewer_camera_look_at(self.viewer, env, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, self.viewer_render_collision)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    def save_replay_state_dict(self, path):
        if not self.save_replay:
            return
        replay_state_dict = {
            'actions': self.replay_actions,
            'root_states': self.replay_root_states,
            'dof_states': self.replay_dof_states,
        }
        torch.save(replay_state_dict, path)
        print('Replay saved', path)

    def load_replay_state_dict(self, path):
        sd = torch.load(path)
        self.replay_actions = sd['actions'].to(self.device)
        self.replay_root_states = sd['root_states'].to(self.device)
        self.replay_dof_states = sd['dof_states'].to(self.device)

    def replay(self):
        self.replay_episode_pt = -1
        self.replay_action_pt = -1
        self.replay_mode = True
        self.save_replay = False
        self.reset()
        while True:
            if self.replay_mode is False:
                break
            if self.replay_episode_pt >= len(self.replay_root_states):
                break
            obs_dict, rew_buf, reset_buf, extras = self.step(None)

    def compute_observations(self):
        pass

    def compute_reward(self):
        pass
