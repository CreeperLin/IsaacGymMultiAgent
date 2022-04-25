from gym import spaces
import numpy as np
import torch
from functools import partial
from argparse import Namespace
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import task_registry
from igma.utils.registry import register


def get_default_env_args(overrides=None):
    dct = {
        'num_envs': None,
    }
    if overrides is not None:
        dct.update(dict(overrides))
    args = Namespace()
    args.__dict__.update(dct)
    return args


def update_cfg(dest, src):
    assert not isinstance(dest, dict)
    if isinstance(src, Namespace):
        dct = src.__dict__
    elif isinstance(src, dict):
        dct = src
    for k, v in dct.items():
        if isinstance(v, (dict, Namespace)):
            ns = getattr(dest, k, Namespace())
            setattr(dest, k, ns)
            update_cfg(ns, v)
        else:
            setattr(dest, k, v)


def patch_env(env):
    env.observation_space = spaces.Box(*[(-1)**(i+1) * np.Inf * np.ones(env.num_obs) for i in range(2)])
    env.num_states = env.num_privileged_obs or 0
    if env.num_privileged_obs:
        env.state_space = spaces.Box(*[(-1)**(i+1) * np.Inf * np.ones(env.num_privileged_obs) for i in range(2)])
    env.action_space = spaces.Box(*[(-1)**(i+1) * np.ones(env.num_actions) for i in range(2)])
    fn_step = env.step
    fn_reset = env.reset
    env._ori_step = fn_step
    env._ori_reset = fn_reset

    def new_step(*args, **kwargs):
        ret = fn_step(*args, **kwargs)
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = ret
        obs_dict = {'obs': obs_buf, 'states': privileged_obs_buf}
        return obs_dict, rew_buf, reset_buf, extras

    def new_reset(*args, **kwargs):
        self = env
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        zero_action = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        # obs, privileged_obs, _, _, _ = self.step(zero_action)
        # return obs, privileged_obs
        obs_dict, _, _, _ = self.step(zero_action)
        return obs_dict

    env.step = new_step
    env.reset = new_reset


def debox(obj):
    if isinstance(obj, (dict, list, tuple, str, int, float, bytes)):
        return obj
    return {k: debox(getattr(obj, k)) for k in dir(obj) if not k.startswith('_')}


def legged_gym_builder(name, *args, **kwargs):
    env_cfg, _ = task_registry.get_cfgs(name=name)
    cfg = kwargs.pop('cfg', None)
    print(args, kwargs, cfg)
    ovr_args = get_default_env_args(cfg)
    update_cfg(env_cfg, cfg)
    ovr_args.__dict__.update(kwargs)
    ovr_args.physics_engine = {'flex': gymapi.SIM_FLEX, 'physx': gymapi.SIM_PHYSX}[ovr_args.physics_engine]
    print('args', ovr_args)
    env, env_cfg = task_registry.make_env(name=name, *args, args=ovr_args, env_cfg=env_cfg)
    print('env', debox(env_cfg))
    patch_env(env)
    return env


def import_registry():
    for name in task_registry.task_classes:
        reg_name = 'LeggedGym' + ''.join([n.title() for n in name.split('_')])
        print(f'importing legged_gym env {name} as {reg_name}')
        builder = partial(legged_gym_builder, name)
        register(builder, reg_name)
        register(builder, name)
        # register(task_registry.task_classes[name], name)


import_registry()
