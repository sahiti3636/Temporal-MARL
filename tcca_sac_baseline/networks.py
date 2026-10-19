import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor Network (Discrete SAC): 
    Outputs categorical action distribution probabilities and log probabilities.
    """
    def __init__(self, obs_dim, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=-1)
        # Add epsilon to prevent NaN loss conditions during optimization
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

class TwinCentralCritics(nn.Module):
    """
    Twin Centralized Critic Networks: 
    Q1 and Q2 evaluated natively over the state concatenations exactly.
    """
    def __init__(self, state_dim, n_agents, n_actions):
        super(TwinCentralCritics, self).__init__()
        input_dim = state_dim + (n_agents - 1) * n_actions
        
        # Twin 1
        self.q1_fc1 = nn.Linear(input_dim, 64)
        self.q1_fc2 = nn.Linear(64, 64)
        self.q1_fc3 = nn.Linear(64, n_actions)
        
        # Twin 2
        self.q2_fc1 = nn.Linear(input_dim, 64)
        self.q2_fc2 = nn.Linear(64, 64)
        self.q2_fc3 = nn.Linear(64, n_actions)

    def forward(self, state, other_actions_onehot):
        x = torch.cat([state, other_actions_onehot], dim=-1)
        
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2
