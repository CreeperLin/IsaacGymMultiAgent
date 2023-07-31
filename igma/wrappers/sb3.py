from typing import Any, List, Optional, Sequence, Type, Union

import torch
import gym
import gym.spaces
try:
    import gymnasium
except ImportError:
    gymnasium = None
import numpy as np
import torch

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from isaacgymenvs.tasks.base.vec_task import VecTask
from .transformed import TransformedVecTask


def patch_space(space):
    if gymnasium is None:
        return space
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)


class IGMAVecEnv(VecEnv):

    def __init__(self, env: VecTask, **kwargs):
        t_env = TransformedVecTask(env, **kwargs)
        self.t_env = t_env
        self.env = env
        VecEnv.__init__(self, env.num_environments, t_env.observation_space, t_env.action_space)
        self.actions = None
        self.metadata = getattr(env, 'metadata', None)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = torch.from_numpy(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.t_env.step(self.actions)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [seed for _ in self.num_envs]

    def reset(self) -> VecEnvObs:
        return self.t_env.reset()

    def close(self) -> None:
        getattr(self.env, 'close', lambda: None)()

    def get_images(self) -> Sequence[np.ndarray]:
        return self.env.render(mode="rgb_array")

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        # return self.env.render(mode=mode)
        return self.env.render()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        attr = getattr(self.env, attr_name)
        return None if attr is None else [attr[i] for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        attr = getattr(self.env, attr_name)
        if attr is None:
            return
        for i in indices:
            attr[i] = value

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        attr = getattr(self.env, method_name)
        return [attr(i, *method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(self.env, wrapper_class) for _ in indices]
