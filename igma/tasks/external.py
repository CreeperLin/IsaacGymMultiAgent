import importlib
from typing import Dict, Any, Tuple
from igma.utils.registry import register
from isaacgymenvs.tasks.base.vec_task import Env
import torch


def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    return obj


class External(Env):

    def __init__(
        self,
        entrypoint=None,
        entry_args=None,
        entry_kwargs=None,
        cfg=None,
        sim_device=None,
        graphics_device_id=None,
        headless=None,
    ) -> None:
        super().__init__(cfg, sim_device, graphics_device_id, headless)
        if cfg is not None:
            entrypoint = entrypoint if entrypoint is not None else cfg.get('entrypoint')
            entry_args = entry_args if entry_args is not None else cfg.get('entry_args')
            entry_kwargs = entry_kwargs if entry_kwargs is not None else cfg.get('entry_kwargs')
        mod_name, fn_name = entrypoint.split(':')
        mod = importlib.import_module(mod_name)
        init_fn = getattr(mod, fn_name)
        ctx = init_fn(*(entry_args or []), **(entry_kwargs or {}))
        # self.ctx = ctx
        self.action_fn, self.step_fn, self.obs_fn, self.rew_fn, self.done_fn, self.info_fn, self.reset_fn = ctx

    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""
        pass

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        self.action_fn(to_device(actions, self.device))
        self.step_fn()
        return (
            to_device(ret, self.rl_device) for ret in (self.obs_fn(), self.rew_fn(), self.done_fn(), self.info_fn()))

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        self.reset_fn(None)
        return to_device(self.obs_fn(), self.rl_device)

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """
        self.reset_fn(env_ids)


register(External)
