import random

class ReplayBuffer:
    """
    Custom Off-Policy Buffer supporting exact contiguous k-step trajectory segments.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, trajectory):
        """
        Stores full episode trajectories cleanly allowing arbitrary k-step bounds internally.
        trajectory step formulation: (obs_list, state, actions, reward, next_obs_list, next_state, done)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, k):
        """
        Samples contiguous k-step horizon segments respecting episode cutoffs.
        Returns a batch configuration of segment lists.
        """
        segments = []
        for _ in range(batch_size):
            ep = random.choice(self.buffer)
            ep_len = len(ep)
            start_idx = random.randint(0, ep_len - 1)
            end_idx = min(ep_len, start_idx + k)
            segments.append(ep[start_idx:end_idx])
        return segments

    def __len__(self):
        return len(self.buffer)
