import yaml
import numpy as np
import igma.tasks.all  # noqa: F401
from igma.utils.paths import get_cfg_dir
from igma.utils.registry import make
from igma.utils.omegaconf import register_resolvers
from igma.wrappers.sb3 import IGMAVecEnv
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import stable_baselines3
try:
    import hydra
except ImportError:
    hydra = None


def main(cfg):
    env = make(
        cfg['task_name'],
        cfg=omegaconf_to_dict(cfg['task']),
        rl_device=cfg['rl_device'],
        sim_device=cfg['sim_device'],
        graphics_device_id=cfg['graphics_device_id'],
        headless=cfg['headless'],
        virtual_screen_capture=cfg['virtual_screen_capture'],
        force_render=cfg['force_render'],
    )
    env = IGMAVecEnv(env)
    train_cfg = cfg['train']
    model_cls = getattr(stable_baselines3, train_cfg['model_cls'])
    model = model_cls(env=env, **train_cfg['model_args'])
    model.learn(**train_cfg['learn'])
    model.save('./ckpt.zip')

    for _ in range(3):
        obs = env.reset()
        rew = 0
        recorded_dones = np.zeros(env.num_envs)
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            recorded_dones = np.logical_or(recorded_dones, done)
            rew += np.mean(reward * np.logical_not(recorded_dones))
            if np.all(recorded_dones):
                print('mean reward', rew)
                break
    env.close()


if __name__ == "__main__":
    if hydra is None:
        main(yaml.safe_load_all(open('./config.yml', 'r')))
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path=get_cfg_dir())(main)()
