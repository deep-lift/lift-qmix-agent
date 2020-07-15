"""
State and Observations
At each timestep, agents receive local observations drawn within their field of view.
This encompasses information about the map within a circular area around each unit and with a radius equal to the sight range.
The sight range makes the environment partially observable from the standpoint of each agent.
Agents can only observe other agents if they are both alive and located within the sight range.
Hence, there is no way for agents to determine whether their teammates are far away or dead.
The feature vector observed by each agent contains the following attributes for both allied and enemy units
within the sight range: distance, relative x, relative y, health, shield, and unit_type 1.
Shields serve as an additional source of protection that needs to be removed before any damage can be done
to the health of units. All Protos units have shields, which can regenerate if no new damage is dealt
(units of the other two races do not have this attribute). In addition, agents have access to the last actions
of allied units that are in the field of view.

Lastly, agents can observe the terrain features surrounding them; particularly, the values of eight points
at a fixed radius indicating height and walkability.
The global state, which is only available to agents during centralised training, contains information about
all units on the map. Specifically, the state vector includes the coordinates of all agents relative to the
centre of the map, together with unit features present in the observations.

Additionally, the state stores the energy of Medivacs and cooldown of the rest of allied units,
which represents the minimum delay between attacks. Finally, the last actions of all agents are
attached to the central state.
All features, both in the state as well as in the observations of individual agents, are normalised by
their maximum values. The sight range is set to 9 for all agents.
"""

import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from argslist import *


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        # self.ep_limit = args.ep_limit
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('init rollout worker')

    def generate_episode_light(self, ep_num=None, evaluate=False):
        o, u, u_onehot, r, s, terminate, padded = [], [], [], [], [], [], []
        d = False
        obs = self.env.reset()
        if RENDER:
            self.env.render()

        terminated = False
        step = 0
        ep_reward = 0
        epsilon = 0 if evaluate else self.epsilon

        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if ep_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        plot_cnt_per_actions = np.asarray([0] * N_ACTION)

        while not d:
            action = self.agents.choose_vanilla_action(obs, epsilon, evaluate)
            plot_cnt_per_actions[action] += 1  # 여기에 액션의 출력과 요청 에이전트 수를 기록하기 위함
            obs, reward, d, _ = self.env.step(action)
            # print(f'step : {step}, reward : {reward}, d : {d}')

            o.append(obs)
            r.append(reward)
            terminate.append(d)
            ep_reward += reward
            step += 1
            padded.append([0.])
            u.append(action)

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        o.append(obs)
        o_next = o[1:]
        o = o[:-1]

        for i in range(step,
                       self.args.max_episode_steps):
            o.append(np.zeros(self.obs_space))
            r.append(0)
            o_next.append(np.zeros(self.obs_space))
            terminate.append(1)
            padded.append([1.])
            u.append(0)

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       terminated=terminate.copy(),
                       padded=padded.copy()
                       )

        for key in episode.keys():
            # if key == 'r' or key == 'terminate':
            #     episode[key] = np.array([episode[key]])
            # else:
                episode[key] = np.array([episode[key]])

        if not evaluate:
            self.epsilon = epsilon

        return episode, ep_reward, plot_cnt_per_actions, epsilon

    def generate_episode(self, ep_num=None, evaluate=False):
        o, u, u_onehot, r, s, terminate, padded = [], [], [], [], [], [], []
        self.env.reset()
        if RENDER:
            self.env.render()

        terminated = False
        step = 0
        ep_reward = 0
        last_action = np.zeros((self.args.num_agents, self.args.num_actions))

        if self.agents.policy.name == 'qmix':
            self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon

        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if ep_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        requested_agents = np.asarray([True] * N_AGENTS)
        plot_ep_requested_agents = np.asarray([0] * N_AGENTS)
        plot_ep_cnt_requested_agent = np.asarray([0] * N_AGENTS)
        plot_cnt_per_actions = np.asarray([0] * N_ACTION)

        while not terminated:
            plot_ep_requested_agents[requested_agents] += 1  # 에이전트 액션별 카운트 (네트워크에서 출력한 에이전트의 액션 빈도)
            count_of_requested_agents = 0  # 요청 에이전트 수 기록
            for b in requested_agents:
                if b: count_of_requested_agents += 1
            plot_ep_cnt_requested_agent[count_of_requested_agents - 1] += 1

            obs = self.env.get_obs()
            state = self.env.get_state()

            actions, actions_onehot = [], []

            for agent_id in range(self.num_agents):
                # todo : last action을 onehot으로 넣고 있는데 사용 여부 확인해서 필요없음 제거
                action = self.agents.choose_action(obs[agent_id], state, last_action[agent_id], agent_id, epsilon, evaluate)

                if requested_agents[agent_id]:
                    action_onehot = np.zeros(self.args.num_actions)
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    last_action[agent_id] = action_onehot
                    plot_cnt_per_actions[action] += 1  # 여기에 액션의 출력과 요청 에이전트 수를 기록하기 위함
                else:
                    action_onehot = np.zeros(self.args.num_actions)
                    action_onehot[action] = 0
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    last_action[agent_id] = action_onehot

            reward, terminated, requested_agents = self.env.step_split(actions)
            terminated = all(terminated)

            # todo : 개별 리워드의 합으로 global reward 계산하나, 개별 리워드로 갈경우 에이전트별 합산하는 로직 재구성 필요
            additional_reward = 0
            while not any(requested_agents) and not terminated:
                rr, terminated, requested_agents = self.env.step_split([0] * N_AGENTS)
                terminated = all(terminated)
                additional_reward += np.sum(rr)
                # print(f'step : {step}, reward : {additional_reward}, termniated : {terminated}, requested_agents: {requested_agents}')
            reward = np.sum(reward)  
            reward += np.sum(additional_reward)

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.num_agents, 1]))
            u_onehot.append(actions_onehot)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            ep_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        for i in range(step, self.args.max_episode_steps):
            o.append(np.zeros((self.num_agents, self.obs_space)))
            u.append(np.zeros([self.num_agents, 1]))
            s.append(np.zeros(self.state_space))
            r.append([0.])
            o_next.append(np.zeros((self.num_agents, self.obs_space)))
            s_next.append(np.zeros(self.state_space))
            u_onehot.append(np.zeros((self.num_agents, self.num_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, ep_reward, plot_cnt_per_actions, plot_ep_requested_agents, plot_ep_cnt_requested_agent, epsilon
