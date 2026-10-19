import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from networks import Actor, CentralCritic

class TCCAAgent:
    """
    Temporal Counterfactual Credit Assignment (TCCA) architecture framework handler.
    Contains strictly configurable logic using k-step returns evaluating dynamic counterfactuals.
    """
    def __init__(self, n_agents, obs_dim, state_dim, n_actions, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01, k=5):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actors = [Actor(obs_dim, n_actions).to(self.device) for _ in range(n_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        
        self.critic = CentralCritic(state_dim, n_agents, n_actions).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_actions(self, obs_list):
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                probs = self.actors[i](obs_tensor)
                dist = torch.distributions.Categorical(probs)
                actions.append(dist.sample().item())
        return actions

    def update(self, trajectories):
        if not trajectories:
            return 0.0
            
        critic_loss_total = 0.0
        updates = 0
        
        for ep in trajectories:
            ep_len = len(ep)
            
            states = torch.FloatTensor(np.array([step[1] for step in ep])).to(self.device) # [T, state_dim]
            actions = torch.LongTensor(np.array([step[2] for step in ep])).to(self.device) # [T, n_agents]
            rewards = torch.FloatTensor(np.array([step[3] for step in ep])).to(self.device) # [T]
            dones = torch.FloatTensor(np.array([step[6] for step in ep])).to(self.device) # [T]
            obs_batch = [step[0] for step in ep]
            
            for agent_idx in range(self.n_agents):
                # 1. Structure Q Values & Probabilities
                other_actions = torch.cat([actions[:, :agent_idx], actions[:, agent_idx+1:]], dim=1)
                other_actions_onehot = F.one_hot(other_actions, num_classes=self.n_actions).view(ep_len, -1).float()
                
                q_values = self.critic(states, other_actions_onehot) # [T, n_actions]
                
                obs_tensor = torch.FloatTensor(np.array([o[agent_idx] for o in obs_batch])).to(self.device)
                probs = self.actors[agent_idx](obs_tensor) # [T, n_actions]
                
                # 2. Counterfactual Baseline B_i(s_t) Computation
                baseline = torch.sum(probs * q_values.detach(), dim=1) # [T]
                
                # V(s_{t+k}) maps exactly over standard baseline calculations natively formulated 
                # strictly on temporal trajectory execution offsets
                V_val = baseline.clone().detach() # [T]
                
                # 3. K-Step Temporal Return logic extraction (G_t Formulation)
                G = torch.zeros(ep_len).to(self.device)
                for t in range(ep_len):
                    g_t = 0
                    discount = 1.0
                    terminated = False
                    
                    for l in range(self.k):
                        if (t + l) >= ep_len:
                            terminated = True
                            break
                            
                        g_t += discount * rewards[t + l].item()
                        
                        if dones[t + l] == 1.0: 
                            terminated = True
                            break
                            
                        discount *= self.gamma
                        
                    # Standard execution handling sequence bounds (bootstrap solely if not sequence completed natively)
                    if not terminated and (t + self.k) < ep_len:
                        g_t += discount * V_val[t + self.k].item()
                        
                    G[t] = g_t
                    
                # 4. Critic MSE Optimization
                q_taken = q_values.gather(1, actions[:, agent_idx].unsqueeze(1)).squeeze(1) # [T]
                critic_loss = F.mse_loss(q_taken, G)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_loss_total += critic_loss.item()
                updates += 1
                
                # 5. TCCA Advantage Calculation (C_i^t) -> Policy Gradients
                advantage = G - baseline.detach()
                
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions[:, agent_idx])
                entropy = dist.entropy().mean()
                
                actor_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
                
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent_idx].step()
                
        return critic_loss_total / max(1, updates)
