#!/usr/bin/env python3
import yaml
import argparse
import datetime
import os
import pprint

import numpy as np

import igma.tasks.all  # noqa: F401
from igma.wrappers.tianshou import IGMAVectorEnv, NestedVectorReplayBuffer
from igma.utils.registry import make
from igma.utils.paths import get_cfg_dir

import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

try:
    import hydra
    from igma.utils.omegaconf import register_resolvers
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
except ImportError:
    hydra = None

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='A1')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--buffer-size', type=int, default=1048576)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    # parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--start-timesteps", type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=3)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=4096)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--test-env', type=bool, default=False)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true', help='watch the play of pre-trained policy only')

    return parser.parse_args('')


def make_env(cfg, num_env=None):
    if num_env:
        cfg['task']['env']['numEnvs'] = num_env
    return make(
        cfg['task_name'],
        cfg=cfg['task'],
        sim_device=cfg['sim_device'],
        graphics_device_id=cfg['graphics_device_id'],
        headless=cfg['headless'],
    )


def test_sac(cfg):
    args = get_args()
    cfg = omegaconf_to_dict(cfg)
    # env = make_env(cfg, args.training_num + args.test_num)
    env = make_env(cfg, args.training_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    train_envs = IGMAVectorEnv([lambda: (env, range(0, args.training_num))])
    print(len(train_envs))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net_a,
                      args.action_shape,
                      max_action=args.max_action,
                      device=args.device,
                      unbounded=True,
                      conditioned_sigma=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    net_c2 = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(actor,
                       actor_optim,
                       critic1,
                       critic1_optim,
                       critic2,
                       critic2_optim,
                       tau=args.tau,
                       gamma=args.gamma,
                       alpha=args.alpha,
                       estimation_step=args.n_step,
                       action_space=env.action_space)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    buffer = NestedVectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    train_collector.collect(n_step=args.start_timesteps * args.training_num, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_sac'
    log_path = os.path.join(args.logdir, args.task, 'sac', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # test
    test_collector = None
    if args.test_env:
        test_envs = IGMAVectorEnv([lambda: env, range(args.training_num, args.training_num + args.test_num)])
        test_envs.seed(args.seed)
        test_collector = Collector(policy, test_envs)

    if not args.watch:
        # trainer
        result = offpolicy_trainer(policy,
                                   train_collector,
                                   test_collector,
                                   args.epoch,
                                   args.step_per_epoch,
                                   args.step_per_collect,
                                   args.test_num,
                                   args.batch_size,
                                   save_fn=save_fn,
                                   logger=logger,
                                   update_per_step=args.update_per_step,
                                   test_in_train=False)
        pprint.pprint(result)

    # Let's watch its performance!
    if test_collector:
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    if hydra is None:
        test_sac(yaml.safe_load_all(open('./config.yml', 'r')))
    else:
        register_resolvers()
        hydra.main(config_name="config", config_path=get_cfg_dir())(test_sac)()
