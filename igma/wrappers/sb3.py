from typing import Any, List, Optional, Sequence, Type, Union

import torch
import gym
import gym.spaces
try:
    import gymnasium
except ImportError:
    gymnasium = None
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from isaacgymenvs.tasks.base.vec_task import VecTask


def patch_space(space):
    if gymnasium is None:
        return space
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)


class IGMAVecEnv(VecEnv):

    def __init__(self, env: VecTask, use_gymnasium_spaces=False, buf_transform_fn=None):
        self.env = env
        obs_space, act_space = env.observation_space, env.action_space
        if use_gymnasium_spaces:
            obs_space, act_space = patch_space(obs_space), patch_space(act_space)
        VecEnv.__init__(self, env.num_environments, obs_space, act_space)
        self.actions = None
        self.buf_transform_fn = lambda x: x.cpu().numpy() if buf_transform_fn is None else buf_transform_fn
        self.metadata = getattr(env, 'metadata', None)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = torch.as_tensor(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs_dict, rew_buf, reset_buf, extras = self.env.step(self.actions)
        obs_buf = obs_dict['obs']
        obs_buf = self.buf_transform_fn(obs_buf)
        rew_buf = self.buf_transform_fn(rew_buf)
        reset_buf = self.buf_transform_fn(reset_buf)
        if not isinstance(extras, (list, tuple)):
            extras = [extras] * self.num_envs
        return obs_buf, rew_buf, reset_buf, extras

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [seed for _ in self.num_envs]

    def reset(self) -> VecEnvObs:
        obs_dict = self.env.reset()
        obs_buf = obs_dict['obs']
        obs_buf = self.buf_transform_fn(obs_buf)
        return obs_buf

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
