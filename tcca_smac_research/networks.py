import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNActor(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNActor, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, avail_actions):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        # Mask unavailable actions
        q[avail_actions == 0] = -1e10
        # Compute probabilities matching SAC requirements
        pi = F.softmax(q, dim=-1)
        
        return pi, h

class CentralCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralCritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class TwinCentralCritics(nn.Module):
    def __init__(self, input_shape, args):
        super(TwinCentralCritics, self).__init__()
        self.critic1 = CentralCritic(input_shape, args)
        self.critic2 = CentralCritic(input_shape, args)

    def forward(self, inputs):
        q1 = self.critic1(inputs)
        q2 = self.critic2(inputs)
        return q1, q2
