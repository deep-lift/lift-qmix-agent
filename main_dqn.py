from env_args import *
from argslist import *
from agent.agent import Agents
from agent.dqn_event_driven import DQN
from worker import RolloutWorker
import elevator
from util.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

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
env = elevator.ElevatorEnv(SCREEN_WIDTH, SCREEN_HEIGHT, False)
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
        episode, episode_reward, episode_count_per_actions, episode_episode_requested_agents, episode_episode_count_requested_agent, current_epsilon = worker.generate_episode(e)
        plot_count_per_actions += episode_count_per_actions
        plot_episode_requested_agents += episode_episode_requested_agents
        plot_episode_count_requested_agent += episode_episode_count_requested_agent
        plot_episode_rewards.append(episode_reward)
        plot_episode_epsilon.append(current_epsilon)
        episodes.append(episode)
        print(f'{n_episode} episode reward: {episode_reward}, epsilon : {current_epsilon}')

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

        writer.add_scalar('episode rewards', episode_reward, n_episode)
        writer.add_scalar('epsilon', current_epsilon, n_episode)

        # plt.rcParams["figure.figsize"] = (50, 50)
        # plt.rcParams['lines.linewidth'] = 4
        #
        # index1 = ["Action 0", "Action 1", "Action 2"]
        # axes[0, 0].bar(x=index1, height=plot_count_per_actions)
        # axes[0, 0].set_title('Cumulative count over action space')
        # # writer.add_histogram('Cumulative count over action space', episode_count_per_actions)
        #
        # index2 = ["1 Agents", "2 Agents", "3 Agents", "4 Agents"]
        # index2 = [f'{i+1} Agents' for i in range(N_AGENTS)]
        # axes[0, 2].bar(x=index2, height=plot_episode_count_requested_agent)
        # axes[0, 2].set_title('Number of valid agents over episode')
        # # writer.add_histogram('Number of valid agents over episode', episode_episode_requested_agents)
        #
        # index3 = ["Agent 1", "Agent 2", "Agent 3", "Agent 4"]
        # index3 = [f'Agent {i + 1}' for i in range(N_AGENTS)]
        # axes[0, 4].bar(x=index3, height=plot_episode_requested_agents)
        # axes[0, 4].set_title('Requested times of each agent')
        # # writer.add_histogram('Requested times of each agent', episode_episode_count_requested_agent)
        # # figure.tight_layout()
        # plt.savefig(save_path + '/plt_{}.png'.format('dqn'), format='png')
        # np.save(save_path + '/win_rates_{}'.format(1), win_rates)
        # np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)
        #
        # plt.close()

    # if epoch != 0 and epoch % PRINT_INTERVAL == 0:
    #     figure, axes = plt.subplots(nrows=1, ncols=5)

# plt.cla()
# plt.subplot(2, 1, 1)
# plt.plot(range(len(win_rates)), win_rates)
# plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
# plt.ylabel('win_rate')
#
# plt.subplot(2, 1, 2)
# plt.plot(range(len(episode_rewards)), episode_rewards)
# plt.xlabel('epoch*{}'.format(args.evaluate_cycle))
# plt.ylabel('episode_rewards')
#
# plt.savefig(save_path + '/plt_{}.png'.format(1), format='png')
# np.save(save_path + '/win_rates_{}'.format(1), win_rates)
# np.save(save_path + '/episode_rewards_{}'.format(1), episode_rewards)


