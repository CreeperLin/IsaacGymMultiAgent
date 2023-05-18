import yaml
import numpy as np
import igma.tasks.all
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

    tot_steps = 1e6
    model_cls = getattr(stable_baselines3, cfg.get('model_cls', 'PPO'))
    model = model_cls("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=tot_steps)
    model.save('./ckpt.zip')

    for _ in range(3):
        obs = env.reset()
        rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            rew += np.mean(reward * np.logical_not(done))
            if np.all(done):
                print('mean reward', rew)
                break
    env.close()


if __name__ == "__main__":
    if hydra is None:
        main(yaml.safe_load_all(open('./config.yml', 'r')))
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path="./cfg")(main)()
