import numpy as np
import torch


class Agents:
    def __init__(self, args, policy):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.policy = policy
        self.args = args

    # def select_action(state, needed_action_agents):
    #     global steps_done
    #     sample = random.random()
    #     eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    #     steps_done += 1
    #     if sample > eps_threshold:
    #         with torch.no_grad():
    #             # t.max(1) will return largest column value of each row.
    #             # second column on max result is index of where max element was
    #             # found, so we pick action with the larger expected reward.
    #             # print('network')
    #             # return policy_net(state).squeeze(0).max(1)[1]
    #             return policy_net(state).max(1).indices.tolist(), eps_threshold
    #     else:
    #         return [random.randrange(N_ACTION) for i in needed_action_agents], eps_threshold
    #
    # def optimize_model(episode):
    #     if len(memory) < BATCH_SIZE:
    #         return
    #     transitions = memory.sample(BATCH_SIZE)
    #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    #     # detailed explanation). This converts batch-array of Transitions
    #     # to Transition of batch-arrays.
    #     batch = Transition(*zip(*transitions))
    #
    #     # Compute a mask of non-final states and concatenate the batch elements
    #     # (a final state would've been the one after which simulation ended)
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                             batch.next_state)), device=device, dtype=torch.bool).to(device)
    #
    #     non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device).double()
    #     state_batch = torch.stack(batch.state).to(device).double()
    #     action_batch = torch.tensor(np.stack(batch.action)).to(device)
    #     reward_batch = torch.tensor(np.stack(batch.reward)).to(device).double()
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken. These are the actions which would've been taken
    #     # for each batch state according to policy_net
    #     state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1).long())
    #
    #     # Compute V(s_{t+1}) for all next states.
    #     # Expected values of actions for non_final_next_states are computed based
    #     # on the "older" target_net; selecting their best reward with max(1)[0].
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32).double()
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values.detach()
    #     # Compute the expected Q values
    #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #
    #     # Compute Huber loss
    #     loss = F.smooth_l1_loss(state_action_values.squeeze().reshape(-1), expected_state_action_values)
    #     summary.add_scalar('loss/critic_loss', loss.item(), episode)
    #
    #     # Optimize the model
    #     optimizer.zero_grad()
    #     loss.backward()
    #     for param in policy_net.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     optimizer.step()

    def choose_action(self, obs, state, last_action, agent_num, epsilon, evaluate=False):

        if self.policy.name == 'qmix':
            inputs = obs.copy()
            agent_id = np.zeros(self.num_agents)
            agent_id[agent_num] = 1.
            if self.args.last_action:
                inputs = np.hstack((inputs, last_action))
            if self.args.reuse_network:
                inputs = np.hstack((inputs, agent_id))
        elif self.policy.name == 'dqn':
            inputs = np.hstack((obs, state))
        if self.policy.name == 'qmix':
            hidden_state = self.policy.eval_hidden[:, agent_num, :]
            if self.args.cuda:
                hidden_state = hidden_state.cuda()

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()

        if self.policy.name == 'qmix':
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        elif self.policy.name =='dqn':
            q_value = self.policy.eval_dqn_net(inputs)

        if np.random.uniform() < epsilon and not evaluate:
            action = np.random.choice(self.args.num_actions)
        else:
            action = torch.argmax(q_value).data.item()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]

        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.max_episode_steps):
                if terminated[episode_idx , transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        max_episdoe_len = self._get_max_episode_len(batch)

        for key in batch.keys():
            batch[key] = batch[key][:, :max_episdoe_len]

        self.policy.learn(batch, max_episdoe_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)



