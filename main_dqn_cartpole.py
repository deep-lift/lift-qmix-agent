from env_args import *
from argslist import *
from agent.agent import Agents
from agent.dqn import DQN
from worker import RolloutWorker
import elevator
from util.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import gym

writer = SummaryWriter()
current = datetime.today().strftime('%Y%m%d%H%M%S')

plot_episode_rewards = []  # 이건 에피소드 받은 리워드 ( 에이전트 동안 받은 개별 리워드 다 더한 값)
plot_episode_valid_steps = []  # 에피소드별 action 요청이 하나라도 들어온 step 카운트
plot_episode_count_requested_agent = np.asarray([0] * N_AGENTS)  # 에이전트별 요청받은 에이전트 대수 기록
plot_episode_requested_agents = np.asarray([0] * N_AGENTS)
plot_count_per_actions = np.asarray([0] * N_ACTION)
plot_episode_epsilon = []
args = get_common_args()

## change policy as a DQN
args = dqn_args(args)
policy = DQN(args)

agents = Agents(args, policy)
env = gym.make('CartPole-v0')
worker = RolloutWorker(env, agents, args)
buffer = ReplayBuffer(args)

plt.figure()
plt.axis([0, args.n_epoch, 0, 100])
win_rates = []
episode_rewards = []
train_steps = 0

save_path = args.result_dir + '/' + current
os.makedirs(save_path, exist_ok=True)

n_episode = 0
for epoch in range(args.n_epoch):
    episodes = []
    for e in range(args.n_episodes):
        n_episode = n_episode + 1
        episode, episode_reward, episode_count_per_actions, current_epsilon = worker.generate_episode_light(e)
        # plot_count_per_actions += episode_count_per_actions
        # plot_episode_rewards.append(episode_reward)
        # plot_episode_epsilon.append(current_epsilon)
        episodes.append(episode)
        print(f'{n_episode} episode reward: {episode_reward}, epsilon : {current_epsilon}')

    episode_batch = episodes[0]
    episodes.pop(0)
    for episode in episodes:
        for key in episode_batch.keys():
            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

    buffer.store_episode_vanilla(episode_batch)
    for train_step in range(args.train_steps):
        mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
        agents.train(mini_batch, train_steps)
        train_steps += 1
        writer.add_scalar('episode rewards', episode_reward, n_episode)
        writer.add_scalar('epsilon', current_epsilon, n_episode)
