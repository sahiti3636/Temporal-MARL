import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from networks import Actor, TwinCentralCritics

class TCCASACAgent:
    """
    TCCA algorithm dynamically integrated with Discrete Soft Actor-Critic handling.
    Executes off-policy training dynamically configuring minimal entropy.
    """
    def __init__(self, n_agents, obs_dim, state_dim, n_actions, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, alpha=0.005, tau=0.005, k=3):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha  # Minimum requested temperature constant avoiding auto-tuning
        self.tau = tau
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actors = [Actor(obs_dim, n_actions).to(self.device) for _ in range(n_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        
        self.critic = TwinCentralCritics(state_dim, n_agents, n_actions).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def get_actions(self, obs_list):
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                probs, _ = self.actors[i](obs_tensor)
                dist = torch.distributions.Categorical(probs)
                actions.append(dist.sample().item())
        return actions

    def soft_update(self):
        """Standard SAC polyak averaging updates."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, batch_segments):
        if not batch_segments: 
            return 0.0, 0.0
            
        batch_size = len(batch_segments)
        
        # t0 refers to the starting execution position in an n-step bounded environment representation.
        t0_steps = [seg[0] for seg in batch_segments]
        
        states = torch.FloatTensor(np.array([s[1] for s in t0_steps])).to(self.device)
        actions = torch.LongTensor(np.array([s[2] for s in t0_steps])).to(self.device)
        obs_batch = [s[0] for s in t0_steps]
        
        critic1_loss_total, critic2_loss_total = 0.0, 0.0
        
        for agent_idx in range(self.n_agents):
            other_actions = torch.cat([actions[:, :agent_idx], actions[:, agent_idx+1:]], dim=1)
            other_actions_onehot = F.one_hot(other_actions, num_classes=self.n_actions).view(batch_size, -1).float()
            
            q1_vals, q2_vals = self.critic(states, other_actions_onehot)
            q1_taken = q1_vals.gather(1, actions[:, agent_idx].unsqueeze(1)).squeeze(1)
            q2_taken = q2_vals.gather(1, actions[:, agent_idx].unsqueeze(1)).squeeze(1)
            
            # Counterfactual Baseline B_i(s_t) Computation = sum(pi * min(Q1, Q2))
            obs_tensor = torch.FloatTensor(np.array([o[agent_idx] for o in obs_batch])).to(self.device)
            probs, log_probs = self.actors[agent_idx](obs_tensor)
            
            min_q_vals = torch.min(q1_vals, q2_vals)
            baseline = torch.sum(probs.detach() * min_q_vals.detach(), dim=1) # [batch_size]
            
            # Formulating k-step Temporal horizon evaluation G_t inherently utilizing Soft Value constraints
            G = torch.zeros(batch_size).to(self.device)
            with torch.no_grad():
                for b in range(batch_size):
                    seg = batch_segments[b]
                    seg_len = len(seg)
                    g_val = 0.0
                    discount = 1.0
                    terminated = False
                    
                    for l in range(seg_len):
                        r = seg[l][3]
                        d = seg[l][6]
                        g_val += discount * r
                        if d == 1.0:
                            terminated = True
                            break
                        discount *= self.gamma
                        
                    # Validating soft bootstrap execution tracking
                    if not terminated and seg_len == self.k:
                        next_state = seg[-1][5]
                        next_obs_list = seg[-1][4]
                        
                        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        next_obs_t = torch.FloatTensor(next_obs_list[agent_idx]).unsqueeze(0).to(self.device)
                        
                        n_act = []
                        for i in range(self.n_agents):
                            not_i = torch.FloatTensor(next_obs_list[i]).unsqueeze(0).to(self.device)
                            p, _ = self.actors[i](not_i)
                            n_act.append(torch.distributions.Categorical(p).sample())
                            
                        n_act = torch.stack(n_act, dim=1)
                        n_other = torch.cat([n_act[:, :agent_idx], n_act[:, agent_idx+1:]], dim=1)
                        n_other_1h = F.one_hot(n_other, num_classes=self.n_actions).view(1, -1).float()
                        
                        t_q1, t_q2 = self.target_critic(next_state_t, n_other_1h)
                        min_t_q = torch.min(t_q1, t_q2)
                        
                        n_probs, n_log_probs = self.actors[agent_idx](next_obs_t)
                        # Soft Value formulation handling intrinsic stochastic environments
                        v_next = torch.sum(n_probs * (min_t_q - self.alpha * n_log_probs), dim=1).item()
                        g_val += discount * v_next
                        
                    G[b] = g_val
                    
            # Twin Critic Parameter Iteration Update
            critic1_loss = F.mse_loss(q1_taken, G)
            critic2_loss = F.mse_loss(q2_taken, G)
            
            self.critic_optimizer.zero_grad()
            (critic1_loss + critic2_loss).backward()
            self.critic_optimizer.step()
            
            critic1_loss_total += critic1_loss.item()
            critic2_loss_total += critic2_loss.item()
            
            # TCCA Advantage Calculation (C_i^t)
            advantage = G - baseline.detach()
            
            # SAC Formulative Policy Updates targeting minimal entropy representations directly tracking advantages 
            dist = torch.distributions.Categorical(probs)
            log_probs_taken = dist.log_prob(actions[:, agent_idx])
            
            sac_entropy = (probs * self.alpha * log_probs).sum(dim=1)
            actor_loss = sac_entropy.mean() - (log_probs_taken * advantage.detach()).mean()
            
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()
            
        self.soft_update()
        
        return critic1_loss_total / self.n_agents, critic2_loss_total / self.n_agents
