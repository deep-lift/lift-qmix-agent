from env import *
from collections import deque

from worker import RolloutWorker
from agent import Agents
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer

import elevator
from argslist import *
import numpy as np
import random
import os
from datetime import datetime
import seaborn as sns

yyyymmddHHMMSS = datetime.today().strftime('%Y%m%d%H%M%S')

# 재영님 여기에 데이터 기록 해볼께요~~ 플롯 찍어보기 위한 변수들 (평가 + 훈련 구분없이 다 저장하게 함)
plot_episode_rewards = []  # 이건 에피소드 받은 리워드 ( 에이전트 동안 받은 개별 리워드 다 더한 값)
plot_episode_valid_steps = []  # 에피소드별 action 요청이 하나라도 들어온 step 카운트
plot_episode_requested_agents = np.asarray([0] * N_AGENTS)
plot_count_per_actions = np.asarray([0] * N_ACTION)

args = get_common_args()
args = qmix_args(args)

agents = Agents(args)
env = elevator.ElevatorEnv(SCREEN_WIDTH, SCREEN_HEIGHT, False)

worker = RolloutWorker(env, agents, args)
buffer = ReplayBuffer(args)

plt.figure()
plt.axis([0, args.n_epoch, 0, 100])
win_rates = []
episode_rewards = []
train_steps = 0

save_path = args.result_dir + '/' + yyyymmddHHMMSS
os.makedirs(save_path, exist_ok=True)

for epoch in range(args.n_epoch):

    episodes = []
    for e in range(args.n_episodes):
        episode, episode_reward, episode_count_per_actions, episode_episode_requested_agents = worker.generate_episode(e)
        plot_count_per_actions += episode_count_per_actions
        plot_episode_requested_agents += episode_episode_requested_agents
        plot_episode_rewards.append(episode_reward)
        episodes.append(episode)

    episode_batch = episodes[0]
    episodes.pop(0)
    for episode in episodes:
        for key in episode_batch.keys():
            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

    buffer.store_episode(episode_batch)
    for train_step in range(args.train_steps):
        mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
        agents.train(mini_batch, train_steps)
        train_steps += 1

    plt.cla()
    plt.subplot(3, 1, 1)
    index1 = ["Action 0", "Action 1", "Action 2"]
    plt.bar(x=index1, height=plot_count_per_actions)
    plt.xlabel('Action')
    plt.ylabel('Cumulative action count from network output')

    plt.subplot(3, 1, 2)
    index2 = ["1 agents", "2 agents", "3 agents", "4 agents"]
    plt.bar(x=index2, height=plot_episode_requested_agents)
    plt.xlabel('number of valid agents')
    plt.ylabel('Cumulative count of valid agent asking actions')

    plt.subplot(3, 1, 3)
    plt.plot(range(len(plot_episode_rewards)), plot_episode_rewards)
    plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
    plt.ylabel('episode_rewards')

    plt.savefig(save_path + '/plt_{}.png'.format(1), format='png')
    # np.save(save_path + '/win_rates_{}'.format(1), win_rates)
    # np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)

plt.cla()
plt.subplot(2, 1, 1)
plt.plot(range(len(win_rates)), win_rates)
plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
plt.ylabel('win_rate')

plt.subplot(2, 1, 2)
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
plt.ylabel('episode_rewards')

plt.savefig(save_path + '/plt_{}.png'.format(1), format='png')
np.save(save_path + '/win_rates_{}'.format(1), win_rates)
np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)


