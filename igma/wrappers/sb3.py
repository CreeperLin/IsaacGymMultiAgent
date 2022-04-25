from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from isaacgymenvs.tasks.base.vec_task import VecTask


class IGMAVecEnv(VecEnv):

    def __init__(self, env: VecTask):
        self.env = env
        VecEnv.__init__(self, env.num_environments, env.observation_space, env.action_space)
        self.actions = None
        self.metadata = getattr(env, 'metadata', None)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs_dict, rew_buf, reset_buf, extras = self.env.step(self.actions)
        obs_buf = obs_dict['obs']
        # return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
        obs_buf = obs_buf.cpu().numpy()
        rew_buf = rew_buf.cpu().numpy()
        reset_buf = reset_buf.cpu().numpy()
        return obs_buf, rew_buf, reset_buf, extras

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [seed for _ in self.num_envs]

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        return obs

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
