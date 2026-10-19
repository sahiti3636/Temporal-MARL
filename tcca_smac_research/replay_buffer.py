import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, max_ep_len, num_agents, state_shape, obs_shape, n_actions, rnn_hidden_dim):
        self.capacity = capacity
        self.max_ep_len = max_ep_len
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim
        
        self.obs = np.zeros((capacity, max_ep_len + 1, num_agents, obs_shape), dtype=np.float32)
        self.state = np.zeros((capacity, max_ep_len + 1, state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, max_ep_len + 1, num_agents, 1), dtype=np.int64)
        self.actions_onehot = np.zeros((capacity, max_ep_len + 1, num_agents, n_actions), dtype=np.float32)
        self.rewards = np.zeros((capacity, max_ep_len + 1, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, max_ep_len + 1, 1), dtype=np.float32)
        self.avail_actions = np.zeros((capacity, max_ep_len + 1, num_agents, n_actions), dtype=np.float32)
        
        # Store individual hidden states to allow proper recurrent updates from arbitrary starting points
        self.hidden_states = np.zeros((capacity, max_ep_len + 1, num_agents, rnn_hidden_dim), dtype=np.float32)
        
        self.ep_lens = np.zeros(capacity, dtype=np.int32)
        self.current_size = 0
        self.insert_index = 0

    def insert_episode(self, episode_data):
        ep_len = episode_data["ep_len"]
        
        self.obs[self.insert_index, :ep_len + 1] = episode_data["obs"]
        self.state[self.insert_index, :ep_len + 1] = episode_data["state"]
        self.actions[self.insert_index, :ep_len + 1] = episode_data["actions"]
        self.actions_onehot[self.insert_index, :ep_len + 1] = episode_data["actions_onehot"]
        self.rewards[self.insert_index, :ep_len] = episode_data["rewards"]
        self.dones[self.insert_index, :ep_len] = episode_data["dones"]
        self.avail_actions[self.insert_index, :ep_len + 1] = episode_data["avail_actions"]
        self.hidden_states[self.insert_index, :ep_len + 1] = episode_data["hidden_states"]
        self.ep_lens[self.insert_index] = ep_len
        
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_k_step_segments(self, batch_size, k=3):
        # Sample episodes
        ep_idxs = np.random.choice(self.current_size, batch_size, replace=False)
        
        # For each sampled episode, select a start timestep t such that we can extract a segment of len k
        # or up to the end of the episode
        segment_batch = {
            "obs": [],
            "state": [],
            "actions": [],
            "actions_onehot": [],
            "rewards": [],
            "dones": [],
            "avail_actions": [],
            "hidden_states": [], # Only need the first hidden state for GRU unfolding
            "mask": []
        }
        
        for ep_idx in ep_idxs:
            ep_len = self.ep_lens[ep_idx]
            # Choose a random start timestep. Valid range: [0, ep_len-1]
            t_start = np.random.randint(0, max(1, ep_len))
            t_end = min(t_start + k, ep_len)
            
            # Extract data. We want slice [t_start:t_start+k+1]
            # Since t_end <= ep_len, slice length is t_end - t_start + 1 (the +1 is for next state)
            actual_k = t_end - t_start
            
            # Mask to handle segments that reach termination before full k steps
            mask = np.ones((k, 1), dtype=np.float32)
            if actual_k < k:
                mask[actual_k:, :] = 0.0
                
            # Pad sequences if actual_k < k
            pad_len = k - actual_k
            
            obs_slice = self.obs[ep_idx, t_start:t_end + 1]
            state_slice = self.state[ep_idx, t_start:t_end + 1]
            act_slice = self.actions[ep_idx, t_start:t_end + 1]
            act_oh_slice = self.actions_onehot[ep_idx, t_start:t_end + 1]
            rew_slice = self.rewards[ep_idx, t_start:t_end]
            done_slice = self.dones[ep_idx, t_start:t_end]
            avail_slice = self.avail_actions[ep_idx, t_start:t_end + 1]
            h0 = self.hidden_states[ep_idx, t_start]
            
            if pad_len > 0:
                obs_slice = np.pad(obs_slice, ((0, pad_len), (0, 0), (0, 0)), 'constant')
                state_slice = np.pad(state_slice, ((0, pad_len), (0, 0)), 'constant')
                act_slice = np.pad(act_slice, ((0, pad_len), (0, 0), (0, 0)), 'constant')
                act_oh_slice = np.pad(act_oh_slice, ((0, pad_len), (0, 0), (0, 0)), 'constant')
                rew_slice = np.pad(rew_slice, ((0, pad_len), (0, 0)), 'constant')
                done_slice = np.pad(done_slice, ((0, pad_len), (0, 0)), 'constant')
                avail_slice = np.pad(avail_slice, ((0, pad_len), (0, 0), (0, 0)), 'constant')
                
            segment_batch["obs"].append(obs_slice)
            segment_batch["state"].append(state_slice)
            segment_batch["actions"].append(act_slice)
            segment_batch["actions_onehot"].append(act_oh_slice)
            segment_batch["rewards"].append(rew_slice)
            segment_batch["dones"].append(done_slice)
            segment_batch["avail_actions"].append(avail_slice)
            segment_batch["hidden_states"].append(h0)
            segment_batch["mask"].append(mask)
            
        # Convert to numpy then to tensor
        res = {}
        for key, val in segment_batch.items():
            res[key] = torch.tensor(np.array(val))
            if torch.cuda.is_available():
                res[key] = res[key].cuda()
                
        return res
