import types
from typing import Callable, Optional
from isaacgymenvs.tasks.base.vec_task import VecTask
import gym
import torch
try:
    import gymnasium
except ImportError:
    gymnasium = None


def identity(self, x):
    return x


def to_gymnasium_space(self, space):
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    return space


def to_torch(self, x):
    return torch.as_tensor(x)


def to_numpy(self, x):
    return x.cpu().numpy()


def obs_default_tn(self, x):
    return to_numpy(self, x['obs'])


def info_default_tn(self, x):
    if not isinstance(x, (list, tuple)):
        return [x] * self.env.num_envs
    return x


class TransformedVecTask():

    def __init__(self,
                 env: VecTask,
                 space_tn: Optional[Callable] = None,
                 action_tn: Optional[Callable] = to_torch,
                 obs_tn: Optional[Callable] = obs_default_tn,
                 rew_tn: Optional[Callable] = to_numpy,
                 done_tn: Optional[Callable] = to_numpy,
                 info_tn: Optional[Callable] = info_default_tn) -> None:
        self.env = env
        transforms = {
            'space_tn': space_tn,
            'action_tn': action_tn,
            'obs_tn': obs_tn,
            'rew_tn': rew_tn,
            'done_tn': done_tn,
            'info_tn': info_tn,
        }
        for k, tn in transforms.items():
            setattr(self, k, types.MethodType(identity if tn is None else tn, self))
        self.observation_space = self.space_tn(env.observation_space)
        self.action_space = self.space_tn(env.action_space)

    def step(self, action):
        ret = self.env.step(self.action_tn(action))
        return tuple(tn(r) for r, tn in zip(ret, (self.obs_tn, self.rew_tn, self.done_tn, self.info_tn)))

    def reset(self):
        obs = self.env.reset()
        return self.obs_tn(obs)
