import math
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from torch import Tensor


@torch.jit.script
def find_target_nearest_neighbors(
    pos: Tensor,
    target: Tensor,
    k: int = 0,
    max_dist: int = 0
):
    dist = torch.square((pos.unsqueeze(0) - target.unsqueeze(1))).sum(dim=-1)
    if k:
        val, ind = torch.topk(dist, k, largest=False)
        if max_dist:
            ind = ind[val < max_dist]
        return ind
    if max_dist:
        return torch.arange(0, len(pos)).repeat(len(target), 1).to(device=pos.device)[dist < (max_dist ** 2)]


@torch.jit.script
def find_all_nearest_neighbors(
    pos: Tensor,
    num_groups: int,
    k: int,
    max_dist: float = 0.,
) -> Tensor:
    pos = pos.view(num_groups, -1, 3)   # [E, N, 3]
    dist = torch.cdist(pos, pos)    # [E, N, N]
    val, ind = torch.topk(dist, k + 1, largest=False)     # [E, N, K+1]
    if max_dist:
        ind[val > max_dist] = -1
    ind = ind[:, :, 1:]     # [E, N, K]
    return ind


@torch.jit.script
def select_by_env_index(
    x: Tensor,
    ind: Tensor,
    num_envs: int,
    num_agents: int,
) -> Tensor:
    return x.view(num_envs, num_agents, -1)[
        torch.arange(0, num_envs).view(num_envs, 1, 1), ind
    ]   # [E, N, M, S]


@torch.jit.script
def rel_pos_by_env_index(
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
def is_same_team_index(
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
