import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from networks import Actor, CentralCritic

class COMAMultiAgent:
    """
    Handles COMA logic for all agents. 
    Maintains centralized critic and independent actors per agent.
    """
    def __init__(self, n_agents, obs_dim, state_dim, n_actions, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # To ensure fast CPU training, we could force CPU, but default standard behavior 
        # is to use gpu if available. We check it here.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actors (independent one per agent for generality)
        self.actors = [Actor(obs_dim, n_actions).to(self.device) for _ in range(n_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        
        # Centralized Critic
        self.critic = CentralCritic(state_dim, n_agents, n_actions).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_actions(self, obs_list):
        """
        Select actions for all agents based on their local observations.
        Returns selected actions and action probabilities (mostly for debugging tracking).
        """
        actions = []
        action_probs = []
        
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # [1, obs_dim]
                probs = self.actors[i](obs_tensor)
                
                # Sample categorical distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                
                actions.append(action)
                action_probs.append(probs.squeeze(0).cpu().numpy())
                
        return actions, action_probs

    def update(self, rollouts):
        """
        Update actor and critic networks using a collected batch of on-policy rollouts.
        rollouts is a list of tuples: 
        (obs_list, state, actions, shared_reward, next_obs_list, next_state, done)
        """
        if len(rollouts) == 0:
            return 0.0
            
        # Parse rollouts into contiguous batches
        batch_size = len(rollouts)
        
        # state_batch: [batch_size, state_dim]
        state_batch = torch.FloatTensor(np.array([step[1] for step in rollouts])).to(self.device)
        # actions_batch: [batch_size, n_agents]
        actions_batch = torch.LongTensor(np.array([step[2] for step in rollouts])).to(self.device)
        # rewards_batch: [batch_size]
        rewards_batch = torch.FloatTensor(np.array([step[3] for step in rollouts])).to(self.device)
        
        # next observations and states
        next_obs_batch = [step[4] for step in rollouts]
        next_state_batch = torch.FloatTensor(np.array([step[5] for step in rollouts])).to(self.device)
        dones_batch = torch.FloatTensor(np.array([step[6] for step in rollouts])).to(self.device)
        
        critic_loss_total = 0.0
        
        ### 1. Update Critic (TD(0) Targets) ###
        
        # To compute TD(0) targets on-policy we need the *next actions* from the valid policy.
        # So we sample actions on the next_obs batch.
        with torch.no_grad():
            next_actions_batch = []
            for agent_idx in range(self.n_agents):
                # gather next observations for this specific agent
                next_obs_tensor = torch.FloatTensor(np.array([r_batch[agent_idx] for r_batch in next_obs_batch])).to(self.device)
                probs = self.actors[agent_idx](next_obs_tensor)
                dist = torch.distributions.Categorical(probs)
                next_actions_batch.append(dist.sample())
            next_actions_batch = torch.stack(next_actions_batch, dim=1) # [batch_size, n_agents]

        self.critic_optimizer.zero_grad()
        critic_loss = 0
        
        # Keep track of computed q_values for the Actor update to reuse (variance reduction + speed)
        # Detaching here is crucial so actors don't backprop through the critic graph!
        q_values_detached_all_agents = []
        q_taken_detached_all_agents = []
        
        # Critic computes MSE TD(0) sum over all agents
        for agent_idx in range(self.n_agents):
            # Compute current Q
            other_actions = torch.cat([actions_batch[:, :agent_idx], actions_batch[:, agent_idx+1:]], dim=1)
            other_actions_onehot = F.one_hot(other_actions, num_classes=self.n_actions).view(batch_size, -1).float()
            
            # Predict Q for the current state (Outputs Q for all actions)
            q_values = self.critic(state_batch, other_actions_onehot) # [batch_size, n_actions]
            # Gather Q for the actual action taken
            q_taken = q_values.gather(1, actions_batch[:, agent_idx].unsqueeze(1)).squeeze(1) # [batch_size]
            
            # Compute TD(0) Targets for Q
            with torch.no_grad():
                next_other_actions = torch.cat([next_actions_batch[:, :agent_idx], next_actions_batch[:, agent_idx+1:]], dim=1)
                next_other_actions_onehot = F.one_hot(next_other_actions, num_classes=self.n_actions).view(batch_size, -1).float()
                
                next_q_values = self.critic(next_state_batch, next_other_actions_onehot)
                next_q_taken = next_q_values.gather(1, next_actions_batch[:, agent_idx].unsqueeze(1)).squeeze(1)
                
                td_target = rewards_batch + self.gamma * (1.0 - dones_batch) * next_q_taken
            
            # Critic MSE Loss
            critic_loss += F.mse_loss(q_taken, td_target)
            
            # Store detached values for the actor steps
            q_values_detached_all_agents.append(q_values.detach())
            q_taken_detached_all_agents.append(q_taken.detach())
            
        critic_loss.backward()
        self.critic_optimizer.step()
        critic_loss_total = critic_loss.item()
        
        ### 2. Update Actors (COMA Advantage Calculation) ###
        
        for agent_idx in range(self.n_agents):
            self.actor_optimizers[agent_idx].zero_grad()
            
            # Re-run actor on obs batch to get the computation graph with gradients
            obs_tensor = torch.FloatTensor(np.array([step[0][agent_idx] for step in rollouts])).to(self.device)
            probs = self.actors[agent_idx](obs_tensor) # [batch_size, n_actions]
            
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions_batch[:, agent_idx]) # [batch_size]
            entropy = dist.entropy().mean()
            
            # COMA Counterfactual Baseline: \sum_a \pi(a|o) * Q(s, a, u_{-a})
            # Detach probability inside baseline context because baseline acts purely as a variance 
            # reduction constant target. (Backpropping through baseline adds variance).
            baseline = torch.sum(probs.detach() * q_values_detached_all_agents[agent_idx], dim=1) # [batch_size]
            
            # COMA Advantage: Q(s, a_taken, u_{-a}) - Baseline
            advantage = q_taken_detached_all_agents[agent_idx] - baseline # [batch_size]
            
            # Note: We do NOT detach the advantage itself, because it is just a scaled constant tensor 
            # w.r.t the actor parameters.
            actor_loss = -(log_probs * advantage).mean() - self.entropy_coef * entropy
            
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()
            
        return critic_loss_total

    def save_models(self, path="."):
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{path}/actor_{i}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_models(self, path="."):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{path}/actor_{i}.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth", map_location=self.device))

