from typing import Any, Callable, List, Optional, Sequence, Type, Union

import torch
import gym
import gym.spaces
import gymnasium
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from isaacgymenvs.tasks.base.vec_task import VecTask


def patch_space(space):
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)


class IGMAVecEnv(VecEnv):

    def __init__(self, env: VecTask):
        self.env = env
        VecEnv.__init__(self, env.num_environments, patch_space(env.observation_space), patch_space(env.action_space))
        self.actions = None
        self.dones = np.zeros(env.num_environments)
        self.metadata = getattr(env, 'metadata', None)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = torch.from_numpy(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs_dict, rew_buf, reset_buf, extras = self.env.step(self.actions)
        obs_buf = obs_dict['obs']
        obs_buf = self._transform_buf(obs_buf)
        rew_buf = self._transform_buf(rew_buf)
        reset_buf = self._transform_buf(reset_buf)
        if not isinstance(extras, (list, tuple)):
            extras = [extras] * self.env.num_environments
        # print(reset_buf)
        self.dones = np.logical_or(self.dones, reset_buf)
        # print(self.dones)
        return obs_buf, rew_buf, self.dones, extras

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [seed for _ in self.num_envs]

    def _transform_buf(self, buf):
        return buf.cpu().numpy()

    def reset(self) -> VecEnvObs:
        self.dones.fill(0)
        obs_dict = self.env.reset()
        obs_buf = obs_dict['obs']
        return self._transform_buf(obs_buf)

    def close(self) -> None:
        getattr(self.env, 'close', lambda: None)()

    def get_images(self) -> Sequence[np.ndarray]:
        return self.env.render(mode="rgb_array")

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode=mode)

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
