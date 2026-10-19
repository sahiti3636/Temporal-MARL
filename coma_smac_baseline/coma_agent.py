# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# from networks import RNNActor, CentralCritic

# class COMAMACAgent:
#     """
#     Advanced agent architecture integrating exact Counterfactual calculations over GRU sequentially.
#     """
#     def __init__(self, n_agents, obs_dim, state_dim, n_actions, lr_actor=5e-4, lr_critic=5e-4, gamma=0.99, entropy_coef=0.01):
#         self.n_agents = n_agents
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.entropy_coef = entropy_coef
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Identity matrices for explicit topological tracking
#         self.agent_id_eye = torch.eye(n_agents).to(self.device)
        
#         # Construct GRU actor handling integrated execution constraints natively 
#         self.actor = RNNActor(obs_dim, n_actions, n_agents).to(self.device)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
#         self.critic = CentralCritic(state_dim, obs_dim, n_actions, n_agents).to(self.device)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

#     def init_hidden(self, batch_size=1):
#         return torch.zeros(batch_size, self.n_agents, 64).to(self.device)

#     def get_actions(self, obs, last_actions, hidden_states, avail_actions):
#         """
#         Dynamically extracts probabilistic behaviors strictly masking invalid SC2 engine arrays physically.
#         """
#         with torch.no_grad():
#             obs_tensor = torch.FloatTensor(obs).to(self.device)
#             last_act_onehot = F.one_hot(torch.LongTensor(last_actions).to(self.device), num_classes=self.n_actions).float()
            
#             inputs = torch.cat([obs_tensor, last_act_onehot, self.agent_id_eye], dim=-1)
#             q_vals, next_hidden = self.actor(inputs, hidden_states.squeeze(0))
            
#             # Formulate strict algorithmic masking masking out native invalid interactions
#             avail_tensor = torch.FloatTensor(avail_actions).to(self.device)
#             q_vals[avail_tensor == 0] = -1e10
            
#             probs = F.softmax(q_vals, dim=-1)
#             dist = torch.distributions.Categorical(probs)
#             actions = dist.sample()
            
#         return actions.cpu().numpy(), next_hidden.unsqueeze(0)

#     def update(self, batch_episodes):
#         critic_loss_total = 0.0
#         updates = 0
        
#         # Traverse strictly sequentially unrolling targets natively 
#         for ep in batch_episodes:
#             ep_len = len(ep)
            
#             obs = torch.FloatTensor(np.array([step["obs"] for step in ep])).to(self.device) 
#             states = torch.FloatTensor(np.array([step["state"] for step in ep])).to(self.device) 
#             actions = torch.LongTensor(np.array([step["actions"] for step in ep])).to(self.device) 
#             avail_actions = torch.FloatTensor(np.array([step["avail_actions"] for step in ep])).to(self.device)
#             rewards = torch.FloatTensor(np.array([step["reward"] for step in ep])).to(self.device) 
#             last_actions = torch.LongTensor(np.array([step["last_actions"] for step in ep])).to(self.device) 
            
#             # Unroll sequential actor dynamics maintaining chronological dependency implicitly
#             q_vals_list = []
#             h = self.init_hidden(1).squeeze(0)
            
#             # Iterate unrolling accurately
#             for t in range(ep_len):
#                 last_act_onehot = F.one_hot(last_actions[t], num_classes=self.n_actions).float()
#                 inputs = torch.cat([obs[t], last_act_onehot, self.agent_id_eye], dim=-1)
#                 q, h = self.actor(inputs, h)
#                 q_vals_list.append(q)
                
#             q_vals_tensor = torch.stack(q_vals_list, dim=0) # [T, n_agents, n_actions]
            
#             # Align boolean masking exactly configuring gradient bounds
#             q_vals_tensor[avail_actions == 0] = -1e10
#             probs = F.softmax(q_vals_tensor, dim=-1) 
            
#             # Produce precise MC targeting metrics mathematically tracking TD architectures
#             G = torch.zeros(ep_len).to(self.device)
#             ret = 0.0
#             for t in reversed(range(ep_len)):
#                 ret = rewards[t] + self.gamma * ret
#                 G[t] = ret
                
#             # Perform multi-objective execution updates synchronously
#             for i in range(self.n_agents):
#                 other_acts = torch.cat([actions[:, :i], actions[:, i+1:]], dim=1)
#                 other_acts_onehot = F.one_hot(other_acts, num_classes=self.n_actions).view(ep_len, -1).float()
                
#                 agent_id_batch = self.agent_id_eye[i].unsqueeze(0).repeat(ep_len, 1)
                
#                 # Fetch baseline evaluation constraints physically
#                 q_critic = self.critic(states, obs[:, i], other_acts_onehot, agent_id_batch) 
#                 q_taken = q_critic.gather(1, actions[:, i].unsqueeze(1)).squeeze(1) 
                
#                 critic_loss = F.mse_loss(q_taken, G)
                
#                 self.critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 self.critic_optimizer.step()
#                 critic_loss_total += critic_loss.item()
                
#                 # Counterfactual Extraction: B_i(s_t) = sum(pi * Q)
#                 baseline = torch.sum(probs[:, i].detach() * q_critic.detach(), dim=-1)
#                 advantage = q_taken.detach() - baseline
                
#                 dist = torch.distributions.Categorical(probs[:, i])
#                 log_probs = dist.log_prob(actions[:, i])
#                 entropy = dist.entropy().mean()
                
#                 # Execute standard actor boundaries with specific clipping mechanisms requested
#                 actor_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
                
#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0) 
#                 self.actor_optimizer.step()
                
#             updates += 1
            
#         return critic_loss_total / (self.n_agents * max(1, updates))


import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from networks import RNNActor, CentralCritic

class COMAMACAgent:
    """
    Advanced agent architecture integrating exact Counterfactual calculations over GRU sequentially.
    """
    def __init__(self, n_agents, obs_dim, state_dim, n_actions, lr_actor=5e-4, lr_critic=5e-4, gamma=0.99, entropy_coef=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Identity matrices for explicit topological tracking
        self.agent_id_eye = torch.eye(n_agents).to(self.device)
        
        # Construct GRU actor handling integrated execution constraints natively 
        self.actor = RNNActor(obs_dim, n_actions, n_agents).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CentralCritic(state_dim, obs_dim, n_actions, n_agents).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.n_agents, 64).to(self.device)

    def get_actions(self, obs, last_actions, hidden_states, avail_actions):
        """
        Dynamically extracts probabilistic behaviors strictly masking invalid SC2 engine arrays physically.
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            last_act_onehot = F.one_hot(torch.LongTensor(last_actions).to(self.device), num_classes=self.n_actions).float()
            
            inputs = torch.cat([obs_tensor, last_act_onehot, self.agent_id_eye], dim=-1)
            q_vals, next_hidden = self.actor(inputs, hidden_states.squeeze(0))
            
            # Formulate strict algorithmic masking masking out native invalid interactions
            avail_tensor = torch.FloatTensor(avail_actions).to(self.device)
            q_vals[avail_tensor == 0] = -1e10
            
            probs = F.softmax(q_vals, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
        return actions.cpu().numpy(), next_hidden.unsqueeze(0)

    def update(self, batch_episodes):
        critic_loss_total = 0.0
        updates = 0
        
        for ep in batch_episodes:
            ep_len = len(ep)
            
            obs = torch.FloatTensor(np.array([step["obs"] for step in ep])).to(self.device) 
            states = torch.FloatTensor(np.array([step["state"] for step in ep])).to(self.device) 
            actions = torch.LongTensor(np.array([step["actions"] for step in ep])).to(self.device) 
            avail_actions = torch.FloatTensor(np.array([step["avail_actions"] for step in ep])).to(self.device)
            rewards = torch.FloatTensor(np.array([step["reward"] for step in ep])).to(self.device) 
            last_actions = torch.LongTensor(np.array([step["last_actions"] for step in ep])).to(self.device) 
            
            # 1. Forward pass through Actor to get probabilities for the whole episode
            q_vals_list = []
            h = self.init_hidden(1).squeeze(0)
            
            for t in range(ep_len):
                last_act_onehot = F.one_hot(last_actions[t], num_classes=self.n_actions).float()
                inputs = torch.cat([obs[t], last_act_onehot, self.agent_id_eye], dim=-1)
                q, h = self.actor(inputs, h)
                q_vals_list.append(q)
                
            q_vals_tensor = torch.stack(q_vals_list, dim=0) # [T, n_agents, n_actions]
            
            # FIX 1: Use masked_fill instead of inplace assignment to keep the graph intact
            q_vals_tensor = q_vals_tensor.masked_fill(avail_actions == 0, -1e10)
            probs = F.softmax(q_vals_tensor, dim=-1) 
            
            # 2. Calculate Returns (Monte Carlo Targets)
            G = torch.zeros(ep_len).to(self.device)
            ret = 0.0
            for t in reversed(range(ep_len)):
                ret = rewards[t] + self.gamma * ret
                G[t] = ret
                
            # FIX 2: Zero the gradients OUTSIDE the loop
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
                
            # 3. Update Agents (Accumulate gradients in the loop)
            for i in range(self.n_agents):
                # Prepare Critic Inputs
                other_acts = torch.cat([actions[:, :i], actions[:, i+1:]], dim=1)
                other_acts_onehot = F.one_hot(other_acts, num_classes=self.n_actions).view(ep_len, -1).float()
                agent_id_batch = self.agent_id_eye[i].unsqueeze(0).repeat(ep_len, 1)
                
                # --- Critic Update ---
                q_critic = self.critic(states, obs[:, i], other_acts_onehot, agent_id_batch) 
                q_taken = q_critic.gather(1, actions[:, i].unsqueeze(1)).squeeze(1) 
                
                critic_loss = F.mse_loss(q_taken, G)
                critic_loss.backward(retain_graph=True)
                critic_loss_total += critic_loss.item()
                
                # --- Actor Update ---
                baseline = torch.sum(probs[:, i].detach() * q_critic.detach(), dim=-1)
                advantage = q_taken.detach() - baseline
                
                dist = torch.distributions.Categorical(probs[:, i])
                log_probs = dist.log_prob(actions[:, i])
                entropy = dist.entropy().mean()
                
                actor_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
                
                last_loop = (i == self.n_agents - 1)
                actor_loss.backward(retain_graph=not last_loop)
            
            # FIX 3: Step the optimizers ONCE after all agents have accumulated their gradients
            self.critic_optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0) 
            self.actor_optimizer.step()
                
            updates += 1
            
        return critic_loss_total / (self.n_agents * max(1, updates))
