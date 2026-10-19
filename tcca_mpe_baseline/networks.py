import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor Network: Maps standard local observations to a categorical action distribution probabilistically.
    """
    def __init__(self, obs_dim, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class CentralCritic(nn.Module):
    """
    Centralized Critic Network: 
    Accepts global state concatenations alongside one-hot encoded other-agent actions.
    Outputs action-respective Q-Values directly.
    """
    def __init__(self, state_dim, n_agents, n_actions):
        super(CentralCritic, self).__init__()
        input_dim = state_dim + (n_agents - 1) * n_actions
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state, other_actions_onehot):
        x = torch.cat([state, other_actions_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
