import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from argslist import *

class DQNNet(nn.Module):
    def __init__(self, args):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(args.obs_space+args.state_space, args.nn_hidden_dim)
        self.dense2 = nn.Linear(args.nn_hidden_dim, args.nn_hidden_dim)
        self.dense3 = nn.Linear(args.nn_hidden_dim, args.nn_hidden_dim)
        self.dense4 = nn.Linear(args.nn_hidden_dim, args.num_actions)

    def forward(self, x):  # 4 * 46
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x
