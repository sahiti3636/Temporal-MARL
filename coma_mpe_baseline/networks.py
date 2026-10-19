import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor Network: 
    Takes local observation -> Outputs categorical action distribution.
    A simple MLP with 2 hidden layers of 64 units.
    """
    def __init__(self, obs_dim, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Apply softmax over final dimension to get action probabilities
        probs = F.softmax(x, dim=-1)
        return probs

class CentralCritic(nn.Module):
    """
    Centralized Critic Network: 
    Takes the global concatenated state and the actions of *other* agents 
    (one-hot encoded) as input.
    Outputs Q-values for *all* possible actions of the given evaluated agent.
    Wait, to evaluate agent A, it expects A's state and A's NOT-A actions.
    """
    def __init__(self, state_dim, n_agents, n_actions):
        super(CentralCritic, self).__init__()
        # Input: Global state + one-hot actions of (n_agents - 1) other agents
        input_dim = state_dim + (n_agents - 1) * n_actions
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state, other_actions_onehot):
        """
        state: Tensor of shape [batch_size, state_dim]
        other_actions_onehot: Tensor of shape [batch_size, (n_agents - 1) * n_actions]
        Returns Q-values of shape [batch_size, n_actions]
        """
        x = torch.cat([state, other_actions_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
