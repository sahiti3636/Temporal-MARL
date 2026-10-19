import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNActor(nn.Module):
    """
    SMAC Partial Observability GRU Actor Setup.
    Retains chronological execution profiles iteratively.
    """
    def __init__(self, obs_dim, n_actions, n_agents):
        super(RNNActor, self).__init__()
        # Dynamic tracking arrays inherently incorporating standard ID execution paths
        input_dim = obs_dim + n_actions + n_agents
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.rnn = nn.GRUCell(64, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, 64)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class CentralCritic(nn.Module):
    """
    Evaluates True SMAC Global State directly extracted from the core engine alongside local dimensions.
    """
    def __init__(self, state_dim, obs_dim, n_actions, n_agents):
        super(CentralCritic, self).__init__()
        input_dim = state_dim + obs_dim + (n_agents - 1) * n_actions + n_agents
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state, obs, other_actions, agent_id):
        x = torch.cat([state, obs, other_actions, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
