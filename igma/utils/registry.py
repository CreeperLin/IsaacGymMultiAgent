from isaacgymenvs.tasks import isaacgym_task_map


def register(env, name=None):
    name = name or env.__name__
    assert name not in isaacgym_task_map
    isaacgym_task_map[name] = env


def make(name, *args, **kwargs):
    return isaacgym_task_map[name](*args, **kwargs)
