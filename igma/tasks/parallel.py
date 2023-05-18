import numpy as np
import torch
from functools import partial
from torch import Tensor
from igma.utils.registry import make, register


def worker_fn(task_fn, recv_fn, send_fn):
    task = task_fn()
    while True:
        try:
            name, args, kwargs = recv_fn()
        except EOFError:
            break
        if name == '__close__':
            return
        # print(name, args, kwargs)
        attr = getattr(task, name)
        ret = attr(*args, **kwargs) if callable(attr) else attr
        send_fn(ret)


def _init_sequential(task_fn):
    task = task_fn()
    task_ret = None

    def local_recv_fn():
        return task_ret

    def local_send_fn(name, args, kwargs):
        nonlocal task_ret
        attr = getattr(task, name)
        task_ret = attr(*args, **kwargs) if callable(attr) else attr

    return local_send_fn, local_recv_fn, lambda: 0


def _init_threading(task_fn,):
    import threading
    import queue
    remote_q = queue.Queue(1)
    local_q = queue.Queue(1)
    remote_recv_fn = remote_q.get
    remote_send_fn = local_q.put
    local_recv_fn = local_q.get
    local_send_fn = remote_q.put
    th = threading.Thread(target=worker_fn, args=(task_fn, remote_recv_fn, remote_send_fn))
    th.start()

    def close_fn():
        th.join()

    return local_send_fn, local_recv_fn, close_fn


def _init_multiprocessing(task_fn, start_method='forkserver'):
    import multiprocessing
    all_start_methods = multiprocessing.get_all_start_methods()
    start_method = start_method if start_method in all_start_methods else 'spawn'
    ctx = multiprocessing.get_context(start_method)
    local, remote = ctx.Pipe()
    remote_recv_fn = remote.recv
    remote_send_fn = remote.send
    local_recv_fn = local.recv
    local_send_fn = local.send
    p = ctx.Process(target=worker_fn, args=(task_fn, remote_recv_fn, remote_send_fn))
    p.start()

    def close_fn():
        p.join()

    return local_send_fn, local_recv_fn, close_fn


_init_fns = {k.replace('_init_', ''): v for k, v in globals().items() if k.startswith('_init_')}


def _default_override_setter(override, args, kwargs):
    if override is None:
        return args, kwargs
    args = [(override[i] if i in override else a) for i, a in enumerate(args)]
    kwargs = {k: (override[k] if k in override else v) for k, v in kwargs.items()}
    return args, kwargs


class DefaultSplitFn():

    def __init__(self, devices=None) -> None:
        self.devices = devices

    def __call__(self, obj, sizes, extras=None):
        if isinstance(obj, (list, tuple)) and len(obj):
            return zip(*(self(o, sizes, extras) for o in obj))
        if isinstance(obj, dict) and len(obj):
            return [{
                k: sv for k, sv in zip(obj.keys(), vs)
            } for vs in zip(*(self(v, sizes, extras) for v in obj.values()))]
        if isinstance(obj, Tensor):
            tensors = torch.split(obj, split_size_or_sections=(obj.size(0) // sizes), dim=0)
            if self.devices is None:
                return tensors
            return [t.to(device=d) for t, d in zip(tensors, self.devices)]
        if isinstance(obj, np.ndarray):
            return np.split(obj, indices_or_sections=sizes, axis=0)
        return [obj] * sizes


class DefaultMergeFn():

    def __init__(self, device=None) -> None:
        self.device = device

    def __call__(self, objs, extras=None):
        if not isinstance(objs, (list, tuple)):
            return objs
        elm = objs[0]
        if isinstance(elm, (list, tuple)):
            return [self(elms, extras) for elms in zip(*objs)]
        if isinstance(elm, dict):
            return {k: self(vs, extras) for k, vs in zip(elm.keys(), zip(*(e.values() for e in objs)))}
        if isinstance(elm, Tensor):
            device = elm.device if self.device is None else self.device
            if len(elm.size()):
                return torch.cat([o.to(device=device) for o in objs], dim=0)
            return sum([o.to(device=device) for o in objs])
        if isinstance(elm, np.ndarray):
            return np.concatenate(objs, axis=0)
        if not isinstance(elm, str) and hasattr(elm, '__add__'):
            return sum(objs)
        return elm


def _default_task_fn(task_name, task_args=None, task_kwargs=None):
    return make(task_name, *(task_args or []), **(task_kwargs or {}))


_default_wrapped = [
    # 'observation_space',
    # 'action_space',
    # 'num_envs',
    # 'num_acts',
    # 'num_obs',
    'set_viewer',
    'allocate_buffers',
    'set_sim_params_up_axis',
    'create_sim',
    'get_state',
    'pre_physics_step',
    'post_physics_step',
    'step',
    'zero_actions',
    'reset',
    'render',
    '__parse_sim_params',
    'get_actor_params_info',
    'apply_randomizations',
    'noise_lambda',
]
_default_wrapped_table = {}
_default_wrapped_table.update({k: 1 for k in _default_wrapped})


class EnvParallel():

    def __init__(self,
                 *args,
                 task=None,
                 task_fn=None,
                 num_tasks=None,
                 devices=None,
                 overrides=None,
                 override_setter=None,
                 split_fn=None,
                 merge_fn=None,
                 merge_device=None,
                 task_args=None,
                 task_kwargs=None,
                 task_cfg=None,
                 wrapped_table=None,
                 parallel_type='multiprocessing',
                 parallel_kwargs=None,
                 cfg=None,
                 **kwargs):
        if isinstance(cfg, dict):
            cfg.pop('name', None)
            return self.__init__(*args, **cfg, **kwargs)
        init_fn = _init_fns[parallel_type]
        ctxs = []
        if task is not None:
            task_fn = partial(_default_task_fn, task)
        if devices is not None:
            num_tasks = len(devices)
            devices = ['cuda:{}'.format(d) for d in devices]
            overrides = [{'sim_device': d} for d in devices]
        if num_tasks is None:
            raise ValueError('num_tasks required')
        overrides = [None] * num_tasks if overrides is None else overrides
        override_setter = _default_override_setter if override_setter is None else override_setter
        task_kwargs = {} if task_kwargs is None else task_kwargs
        task_args = [] if task_args is None else task_args
        if task_cfg is not None:
            task_kwargs['cfg'] = task_cfg
        task_args.extend(args)
        task_kwargs.update(kwargs)
        for override in overrides:
            _task_args, _task_kwargs = override_setter(override, task_args, task_kwargs)
            ctx = init_fn(partial(task_fn, _task_args, _task_kwargs), **(parallel_kwargs or {}))
            ctxs.append(ctx)
        self.ctxs = ctxs
        self.merge_fn = DefaultMergeFn(device=merge_device) if merge_fn is None else merge_fn
        self.split_fn = DefaultSplitFn(devices=devices) if split_fn is None else split_fn
        self.wrapped_fns = {}
        self.wrapped_table = _default_wrapped_table.copy()
        if wrapped_table is not None:
            self.wrapped_table.update(wrapped_table)

    def close(self):
        for ctx in self.ctxs:
            ctx[0](('__close__', None, None))
            ctx[2]()

    def __getattr__(self, name):
        wrapped_fn = self.wrapped_fns.get(name)
        if wrapped_fn is None:

            def wrapped(*args, **kwargs):
                extras = {'attr': name}
                list_args_kwargs = self.split_fn((args, kwargs), len(self.ctxs), extras)
                for ctx, args_kwargs in zip(self.ctxs, list_args_kwargs):
                    local_send_fn = ctx[0]
                    local_send_fn((name, *args_kwargs))
                rets = []
                for ctx in self.ctxs:
                    local_recv_fn = ctx[1]
                    ret = local_recv_fn()
                    rets.append(ret)
                return self.merge_fn(rets, extras)

            self.wrapped_fns[name] = wrapped
            wrapped_fn = wrapped
        wrapped = self.wrapped_table.get(name)
        if wrapped == 1:
            return wrapped_fn
        if wrapped == -1:
            return None
        return wrapped_fn()


register(EnvParallel)
