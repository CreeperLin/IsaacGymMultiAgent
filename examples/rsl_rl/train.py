import traceback
import igma_tasks
from legged_gym.utils import get_args, task_registry
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from igma.utils.omegaconf import register_resolvers
from igma.utils.registry import make
from igma.wrappers.rsl_rl import patch_env, get_default_train_args
try:
    import hydra
except ImportError:
    hydra = None


def train(args=None):
    if args is None:
        args = get_args()
    cfg = omegaconf_to_dict(args)
    task_name = cfg['task_name']
    is_test = cfg['test']
    env = make(
        task_name,
        cfg=cfg['task'],
        sim_device=cfg['sim_device'],
        graphics_device_id=cfg['graphics_device_id'],
        headless=cfg['headless'],
    )
    patch_env(env)
    train_args = get_default_train_args(cfg['train'])
    train_args.experiment_name = train_args.experiment_name or task_name
    if is_test:
        train_args.resume = True
    ppo_runner, train_cfg = None, None
    for name in [task_name, 'a1']:
        try:
            ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=name, args=train_args)
        except Exception:
            traceback.print_exc()
            continue
        break
    if ppo_runner is None:
        return
    if not is_test:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        return
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    if hydra is None:
        train()
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path="./cfg")(train)()
