"""
Author: Chun-Sheng Wu
For Use of ECEN 743 Final Project @ TAMU
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.atari import boxing_v2
from pettingzoo.classic import tictactoe_v3, chess_v5, go_v5

from net.DRQN import Recurrent

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=152)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=16)
    parser.add_argument("--update-per-step", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--stack-num", type=int, default=1)
    parser.add_argument("--layer-num", type=int, default=3)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[256, 128, 64]
    )
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=1.01,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_env(render_mode=None):
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))
    #return gym.make('CartPole-v1')

def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    # for use of petting.classic
    args.state_shape = (
        observation_space["observation"].shape or observation_space["observation"].n
    )
    args.action_shape = env.action_space.shape or env.action_space.n

    # for use of pettingzoo.atari
    #args.state_shape = env.observation_space.shape or env.observation_space.n
    #args.action_shape = env.action_space.shape or env.action_space.n
    if agent_learn is None:
        # model
        net = Recurrent(args.layer_num, args.state_shape, args.action_shape,
                    args.device).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
        )
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    if agent_opponent is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = RandomPolicy()

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        PrioritizedVectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.training_num,
        alpha=args.alpha,
        beta=args.beta,
        stack_num=args.stack_num,
        ignore_obs_next=True,
        ),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "tic_tac_toe", "ddqn_r")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "tic_tac_toe", "ddqn_r", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[args.agent_id - 1]]


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    result, agent = train_agent(args)
    watch(args, agent)