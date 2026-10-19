import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy

from networks import RNNActor, TwinCentralCritics

class TCCAAgent:
    def __init__(self, args):
        self.args = args
        self.k = args.k_steps
        self.gamma = args.gamma
        self.alpha = args.alpha
        
        # We assume the env runner sets these args based on env
        self.num_agents = args.num_agents
        self.n_actions = args.n_actions
        
        # Actor processing
        self.actor = RNNActor(args.actor_input_dim, args)
        if args.cuda:
            self.actor.cuda()
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.lr)
        
        # Critics
        self.critics = TwinCentralCritics(args.critic_input_dim, args)
        self.target_critics = copy.deepcopy(self.critics)
        if args.cuda:
            self.critics.cuda()
            self.target_critics.cuda()
            
        self.critic_optimizer = Adam(self.critics.parameters(), lr=args.lr)

    def select_actions(self, inputs, hidden_state, avail_actions, test_mode=False):
        pi, next_hidden = self.actor(inputs, hidden_state, avail_actions)
        if test_mode:
            actions = torch.argmax(pi, dim=-1)
        else:
            # Sample from pi
            dist = torch.distributions.Categorical(pi)
            actions = dist.sample()
        return actions, next_hidden

    def update(self, batch):
        # batch shapes:
        # obs: (batch, k+1, num_agents, obs_dim)
        # state: (batch, k+1, state_dim)
        # actions_onehot: (batch, k+1, num_agents, n_actions)
        # actions: (batch, k+1, num_agents, 1)
        # rewards: (batch, k, 1)
        # dones: (batch, k, 1) # done at next state
        # avail_actions: (batch, k+1, num_agents, n_actions)
        # hidden_states: (batch, num_agents, rnn_dim) # at t=0
        # mask: (batch, k, 1)

        batch_size = batch["obs"].shape[0]
        device = batch["obs"].device
        
        # Get actor probabilities for the whole segment length k+1
        # Need to unfold the GRU
        pis = []
        h = batch["hidden_states"]
        for t in range(self.k + 1):
            obs_t = batch["obs"][:, t] # (batch, num_agents, obs_dim)
            avail_actions_t = batch["avail_actions"][:, t]
            
            # Reshape for actor
            obs_in = obs_t.reshape(batch_size * self.num_agents, -1)
            avail_actions_in = avail_actions_t.reshape(batch_size * self.num_agents, -1)
            
            pi_t, h = self.actor(obs_in, h, avail_actions_in)
            pis.append(pi_t.reshape(batch_size, self.num_agents, -1))
            
        pis = torch.stack(pis, dim=1) # (batch, k+1, num_agents, n_actions)
        
        # Compute baseline and Q values
        # Critic input: global_state + other_actions_onehot + agent_id
        agent_ids = torch.eye(self.num_agents, device=device).unsqueeze(0).unsqueeze(0)
        agent_ids = agent_ids.expand(batch_size, self.k + 1, self.num_agents, self.num_agents)
        
        state_expanded = batch["state"].unsqueeze(2).expand(batch_size, self.k + 1, self.num_agents, self.args.state_shape)
        
        q1_vals = []
        q2_vals = []
        target_q1_vals = []
        target_q2_vals = []
        
        for i in range(self.num_agents):
            # Other actions
            other_actions = batch["actions_onehot"].clone() # (batch, k+1, num_agents, n_actions)
            other_actions[:, :, i, :] = 0.0 # Mask out agent i
            other_actions_flat = other_actions.reshape(batch_size, self.k + 1, -1) # (batch, k+1, num_agents * n_actions)
            other_actions_expanded = other_actions_flat.unsqueeze(2).expand(batch_size, self.k + 1, self.num_agents, -1)
            
            # Only evaluating for agent i
            critic_in = torch.cat([state_expanded[:, :, i], other_actions_expanded[:, :, i], agent_ids[:, :, i]], dim=-1)
            
            q1, q2 = self.critics(critic_in)
            q1_vals.append(q1)
            q2_vals.append(q2)
            
            with torch.no_grad():
                tq1, tq2 = self.target_critics(critic_in)
                target_q1_vals.append(tq1)
                target_q2_vals.append(tq2)
                
        # Stack over agents: (batch, k+1, num_agents, n_actions)
        q1_vals = torch.stack(q1_vals, dim=2)
        q2_vals = torch.stack(q2_vals, dim=2)
        target_q1_vals = torch.stack(target_q1_vals, dim=2)
        target_q2_vals = torch.stack(target_q2_vals, dim=2)

        # ----------------
        # 1. TCCA Returns
        # ----------------
        with torch.no_grad():
            pi_k = pis[:, self.k]
            # Soft value bootstrapping at t+k
            min_target_q = torch.min(target_q1_vals[:, self.k], target_q2_vals[:, self.k])
            
            # Soft-value V(s_k) = sum_a pi_k * (Q - alpha * log_pi)
            # Add small epsilon to pi_k to avoid log(0)
            log_pi_k = torch.log(pi_k + 1e-10)
            v_k = (pi_k * (min_target_q - self.alpha * log_pi_k)).sum(dim=-1, keepdim=True) # (batch, num_agents, 1)
            
            # Mask v_k if done at any step before k
            # Actually, to compute k-step return recursively
            returns = v_k.clone() # Start from V_k
            
            G_t_list = []
            for t in reversed(range(self.k)):
                r_n = batch["rewards"][:, t].unsqueeze(1).expand(-1, self.num_agents, -1) # (batch, num_agents, 1)
                d_n = batch["dones"][:, t].unsqueeze(1).expand(-1, self.num_agents, -1) # (batch, num_agents, 1)
                m_n = batch["mask"][:, t].unsqueeze(1).expand(-1, self.num_agents, -1) # valid flag
                
                # If done, subsequent return is 0
                returns = r_n + self.gamma * (1 - d_n) * returns
                returns = returns * m_n
                G_t_list.insert(0, returns.clone())
                
            # We only need G_t at t=0 for the actor update, but we can compute MSE over all t in [0, k-1] for the critic
            G_t = G_t_list[0] # (batch, num_agents, 1)
            
        # ----------------
        # 2. Critic Update
        # ----------------
        actions_t0 = batch["actions"][:, :self.k] # (batch, k, num_agents, 1)
        q1_taken = q1_vals[:, :self.k].gather(3, actions_t0).squeeze(-1) # (batch, k, num_agents)
        q2_taken = q2_vals[:, :self.k].gather(3, actions_t0).squeeze(-1)
        
        target_returns = torch.stack(G_t_list, dim=1).squeeze(-1) # (batch, k, num_agents)
        
        # Mask out padded steps
        valid_mask_critic = batch["mask"].expand(-1, -1, self.num_agents) # (batch, k, num_agents)
        
        critic1_loss = (((q1_taken - target_returns) ** 2) * valid_mask_critic).sum() / valid_mask_critic.sum()
        critic2_loss = (((q2_taken - target_returns) ** 2) * valid_mask_critic).sum() / valid_mask_critic.sum()
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.args.grad_norm_clip)
        self.critic_optimizer.step()
        
        # ----------------
        # 3. Actor Update
        # ----------------
        pi_t0 = pis[:, 0] # (batch, num_agents, n_actions)
        q1_t0 = q1_vals[:, 0].detach() # detach from critic graph
        q2_t0 = q2_vals[:, 0].detach()
        min_q_t0 = torch.min(q1_t0, q2_t0)
        
        # Baseline B_i = sum_a pi_t0 * Q_t0
        baseline = (pi_t0 * min_q_t0).sum(dim=-1, keepdim=True) # (batch, num_agents, 1)
        
        # Advantage C_i = G_t - B_i
        advantage = G_t - baseline # (batch, num_agents, 1)
        
        # SAC loss J = E[C_i - alpha * log_pi]
        # We maximize J. We use REINFORCE for the advantage part: advantage * log_pi
        # For the entropy part, we can compute exact entropy or use the sampled log_pi
        actions_first = batch["actions"][:, 0]
        pi_taken = pi_t0.gather(2, actions_first)
        log_pi_taken = torch.log(pi_taken + 1e-10)
        
        # Policy gradient: A * \nabla log_pi
        # Entropy gradient: -\alpha * \nabla log_pi or we just add alpha * H(\pi)
        entropy = -(pi_t0 * torch.log(pi_t0 + 1e-10)).sum(dim=-1, keepdim=True)
        
        actor_objective = (advantage.detach() * log_pi_taken) + self.alpha * entropy
        
        # Valid mask for actor at t=0
        valid_mask_actor = batch["mask"][:, 0].unsqueeze(2).expand(-1, self.num_agents, -1) # (batch, num_agents, 1)
        actor_loss = -(actor_objective * valid_mask_actor).sum() / valid_mask_actor.sum()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.actor_optimizer.step()
        
        # ----------------
        # 4. Target Networks
        # ----------------
        for param, target_param in zip(self.critics.parameters(), self.target_critics.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()
