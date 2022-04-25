import yaml
import igma.tasks.joust
from igma.utils.registry import make
from igma.utils.omegaconf import register_resolvers
from igma.wrappers.sb3 import IGMAVecEnv
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from stable_baselines3 import PPO, SAC
try:
    import hydra
except ImportError:
    hydra = None


def main(cfg):
    env = make(
        cfg['task_name'],
        cfg=omegaconf_to_dict(cfg['task']),
        sim_device=cfg['sim_device'],
        graphics_device_id=cfg['graphics_device_id'],
        headless=cfg['headless'],
    )
    env = IGMAVecEnv(env)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    tot_steps = 1e6
    model = SAC("MlpPolicy", env, verbose=1)
    # model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=tot_steps)
    model.save('./ckpt.zip')

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    if hydra is None:
        main(yaml.safe_load_all(open('./config.yml', 'r')))
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path="./cfg")(main)()
