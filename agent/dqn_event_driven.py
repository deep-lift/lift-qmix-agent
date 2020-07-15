import torch
import os
from network.dqn_net import DQNNet
from util.replay_buffer import ReplayMemory
from argslist import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:

    def __init__(self, args):
        self.name = 'dqn'
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        input_shape = self.obs_space
        if args.last_action:
            input_shape += self.num_actions
        if args.reuse_network:
            input_shape += self.num_agents
        self.eval_dqn_net = DQNNet(args)
        self.target_dqn_net = DQNNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_dqn_net.cuda()
            self.target_dqn_net.cuda()
        self.memory = ReplayMemory(REPLAY_MEM)
        # todo : check alg !!
        self.model_dir = args.model_dir + '/' + args.alg

        if self.args.load_model:
            if os.path.exists(self.model_dir + '/dqn_net_params.pkl'):
                path_dqn = self.model_dir + '/dqn_net_params.pkl'
                self.eval_qmix_net.load_state_dict(torch.load(path_dqn))
                print('Successfully load the model: {}'.format(path_dqn))
            else:
                raise Exception("No model!")

        self.target_dqn_net.load_state_dict(self.eval_dqn_net.state_dict())
        self.eval_parameters = list(self.eval_dqn_net.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        print('Init DQN')

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_dqn_net.state_dict(), self.model_dir + '/' + num + '_dqn_params.pkl')

    def _get_inputs(self, batch, transition_idx):
        obs, state, obs_next, state_next, u_onehot = batch['o'][:, transition_idx], batch['s'][:, transition_idx], batch['o_next'][:, transition_idx], batch['s_next'][:, transition_idx], batch['u_onehot'][:]

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        #inputs.append(np.hstack((obs, state)))

        # todo : 아예 batch 만들때 필요없는 차원 생성 안하게 만들것
        obs = obs.squeeze()
        obs_next = obs_next.squeeze()
        state = state.squeeze()
        state_next = state_next.squeeze()

        inputs.append(
            torch.tensor(
                np.concatenate((obs, state), axis=-1), dtype=torch.float32, device=device
            )
        )

        inputs_next.append(torch.tensor(
            np.concatenate((obs_next, state_next), axis=-1),
            dtype=torch.float32, device=device)
        )

        if self.args.last_action:
            if transition_idx == 0:  # 첫 경험이라면, 이전 행동을 0 벡터로하십시오
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])

        if self.args.reuse_network:
            # 현재 obs 3 차원 데이터에서 각 차원은 (에피소드 수, 에이전트 수, obs 차원)을 나타내므로 해당 벡터를 dim_1에 직접 추가하십시오.
            # 예를 들어 agent_0 뒤에 (1, 0, 0, 0, 0) 만 추가하십시오.
            # 5 개의 에이전트에서 숫자 0을 의미합니다. agent_0의 데이터는 0 번째 행에 있으므로 추가해야합니다.
            # 상담원 번호는 항등 행렬입니다. 즉, 대각선은 1이고 나머지는 0입니다.
            inputs.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1)  )
            inputs_next.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # obs 중 3 개와 episode_num, episode, self.args.n_agents 에이전트의 데이터를 40 개 (40,96)의 데이터로 결합하려면,
        # 여기에있는 모든 에이전트는 신경망을 공유하므로 각 데이터에는 고유 번호가 수반되므로 여전히 자체 데이터입니다
      
        # todo : 여기로직이 좀 이상한것 같음. 한번 체크해보자
        inputs = torch.cat([x.reshape(episode_num * self.args.num_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.num_agents, -1) for x in inputs_next], dim=1)
      
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            q_eval = self.eval_dqn_net(inputs)
            q_target = self.target_dqn_net(inputs_next)

            q_eval = q_eval.view(episode_num, self.num_agents, -1)
            q_target = q_target.view(episode_num, self.num_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]

        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        o, u, s, r, s_next, o_next, terminated = batch['o'], batch['u'], batch['s'], batch['r'], batch['s_next'], batch['o_next'],  batch['terminated']
        mask = 1 - batch["padded"].float()

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s = s.cuda()
            o = o.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            o_next = o_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]

        o = o.squeeze()
        o_next = o_next.squeeze()
        s = s.squeeze()
        s_next = s_next.squeeze()


        q_total_eval = self.eval_dqn_net(torch.cat((o, s),dim=-1))
        q_total_target = self.target_dqn_net(torch.cat((o_next,s_next), dim=-1))

        # todo : check this torch.repeat_interleave
        # q_total_eval = self.eval_dqn_net(torch.cat((o, torch.repeat_interleave(s, repeats=2, dim=1).unsqueeze(1)),dim=-1))
        # q_total_target = self.target_dqn_net(torch.cat((o_next,torch.repeat_interleave(s_next, repeats=2, dim=1).unsqueeze(1)),dim=-1))

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error

        # 쓸모없는 경험 많기 때문에 평균 직접 사용못하며 실제 평균은 실제 경험 수로 나눔.
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # loss = masked_td_error.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_dqn_net.load_state_dict(self.eval_dqn_net.state_dict())
            print('target network updated..')
