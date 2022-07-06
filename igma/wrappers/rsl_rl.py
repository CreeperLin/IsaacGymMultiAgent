from argparse import Namespace
import torch


def get_default_train_args(overrides=None):
    dct = {
        'resume': False,
        'headless': False,
        'horovod': False,
        'rl_device': "cuda:0",
        'experiment_name': None,
        'run_name': None,
        'load_run': None,
        'checkpoint': None,
        'num_envs': None,
        'seed': 1,
        'max_iterations': None,
    }
    if overrides is not None:
        dct.update(dict(overrides))
    args = Namespace()
    args.__dict__.update(dct)
    return args


def patch_env(env):
    ori_step = getattr(env, '_ori_step', None)
    ori_reset = getattr(env, '_ori_reset', None)
    if ori_step is not None and ori_reset is not None:
        env.step = ori_step
        env.reset = ori_reset
        return

    fn_step = env.step
    fn_reset = env.reset

    def step(*args, **kwargs):
        ret = fn_step(*args, **kwargs)
        obs_dict, rew_buf, reset_buf, extras = ret
        obs_buf = obs_dict['obs']
        privileged_obs_buf = obs_dict.get('states', None)
        return obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras

    def reset(*args, **kwargs):
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        zero_action = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)
        obs, privileged_obs, _, _, _ = env.step(zero_action)
        return obs, privileged_obs

    def get_observations(*args, **kwargs):
        return env.obs_buf

    def get_privileged_observations(*args, **kwargs):
        return env.privileged_obs_buf

    num_privileged_obs = env.num_states or None
    episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # time_out_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if num_privileged_obs:
        privileged_obs_buf = torch.zeros(env.num_envs, num_privileged_obs, device=env.device, dtype=torch.float)
    else:
        privileged_obs_buf = None
    patch_tables = {
        'step': step,
        'reset': reset,
        'get_observations': get_observations,
        'get_privileged_observations': get_privileged_observations,
        'num_privileged_obs': num_privileged_obs,
        'episode_length_buf': episode_length_buf,
        'privileged_obs_buf': privileged_obs_buf,
    }
    for key, val in patch_tables.items():
        setattr(env, key, val)
