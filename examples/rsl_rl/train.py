import os
import traceback
import igma.tasks.joust
import igma.tasks.legged_gym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from igma.utils.omegaconf import register_resolvers
from igma.utils.registry import make
from igma.wrappers.rsl_rl import patch_env, get_default_train_args
try:
    import hydra
except ImportError:
    hydra = None


def debox(obj):
    if isinstance(obj, (dict, list, tuple, str, int, float, bytes)):
        return obj
    return {k: debox(getattr(obj, k)) for k in dir(obj) if not k.startswith('_')}


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
    print('cfg', cfg)
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
    print('train', debox(train_cfg))
    if not is_test:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        return
    # EXPORT_POLICY = True
    # RECORD_FRAMES = False
    # MOVE_CAMERA = False
    # if EXPORT_POLICY:
        # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        # export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # print('Exported policy as jit script to: ', path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # logger = Logger(env.dt)
    # robot_index = 0  # which robot is used for logging
    # joint_index = 1  # which joint is used for logging
    # stop_state_log = 100  # number of steps before plotting states
    # stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1., 1., 0.])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # img_idx = 0

    obs = env.get_observations()
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # if RECORD_FRAMES:
        #     if i % 2:
        #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
        #                                 train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
        #         env.gym.write_viewer_image_to_file(env.viewer, filename)
        #         img_idx += 1
        # if MOVE_CAMERA:
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)

        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        #         }
        #     )
        # elif i == stop_state_log:
        #     logger.plot_states()
        # if 0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes > 0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i == stop_rew_log:
            # logger.print_rewards()


if __name__ == '__main__':
    if hydra is None:
        train()
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path="./cfg")(train)()
