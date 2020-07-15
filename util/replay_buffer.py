import numpy as np
import threading
from argslist import *


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.num_actions = self.args.num_actions
        self.num_agents = self.args.num_agents
        self.state_space = self.args.state_space
        self.obs_space = self.args.obs_space
        self.size = self.args.buffer_size
        self.current_idx = 0
        self.current_size = 0

        if self.num_agents == 1 and not IS_EVENT_DRIVEN:
            self.buffers = {'o': np.empty([self.size, self.args.max_episode_steps, self.obs_space]),
                            'u': np.empty([self.size, self.args.max_episode_steps, 1]),
                            'r': np.empty([self.size, self.args.max_episode_steps, 1]),
                            'o_next': np.empty([self.size, self.args.max_episode_steps, self.obs_space]),
                            'padded': np.empty([self.size, self.args.max_episode_steps, 1]),
                            'terminated': np.empty([self.size, self.args.max_episode_steps, 1])
                            }
        else:
            self.buffers = {'o': np.empty([self.size, self.args.max_episode_steps, self.num_agents, self.obs_space]),
                        'u': np.empty([self.size, self.args.max_episode_steps, self.num_agents, 1]),
                        's': np.empty([self.size, self.args.max_episode_steps, self.state_space]),
                        'r': np.empty([self.size, self.args.max_episode_steps, 1]),
                        'o_next': np.empty([self.size, self.args.max_episode_steps, self.num_agents, self.obs_space]),
                        's_next': np.empty([self.size, self.args.max_episode_steps, self.state_space]),
                        'avail_u': np.empty([self.size, self.args.max_episode_steps, self.num_agents, self.num_actions]),
                        'avail_u_next': np.empty([self.size, self.args.max_episode_steps, self.num_agents, self.num_actions]),
                        'u_onehot': np.empty([self.size, self.args.max_episode_steps, self.num_agents, self.num_actions]),
                        'padded': np.empty([self.size, self.args.max_episode_steps, 1]),
                        'terminated': np.empty([self.size, self.args.max_episode_steps, 1])
                        }
        self.lock = threading.Lock()

    def store_episode_vanilla(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # this source code makes me motivated!!
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = np.expand_dims(episode_batch['u'], axis=-1)
            self.buffers['r'][idxs] = np.expand_dims(episode_batch['r'], axis=-1)  # episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['terminated'][idxs] = np.expand_dims(episode_batch['terminated'], axis=-1)  # episode_batch['terminated']

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # this source code makes me motivated!!
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc

        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]

        return idx


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
