import numpy as np
import pandas as pd
import os
from env_wrapper import SimpleSpreadWrapper
from tcca_sac_agent import TCCASACAgent
from replay_buffer import ReplayBuffer

def train():
    env = SimpleSpreadWrapper(N=3, max_cycles=25)
    # Instantiate explicitly requested fixed alpha TCCA architecture configuration
    agent = TCCASACAgent(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, k=3, alpha=0.005)
    
    buffer = ReplayBuffer(capacity=50000)
    
    num_episodes = 50000
    batch_size = 64
    
    csv_file = "training_logs.csv"
    if os.path.exists(csv_file): 
        os.remove(csv_file)
        
    pd.DataFrame(columns=["Episode", "Reward", "Critic1 Loss", "Critic2 Loss"]).to_csv(csv_file, index=False)
    
    print("Starting Off-Policy TCCA-SAC training locally...")
    
    ep_returns = []
    
    for ep in range(1, num_episodes + 1):
        obs, state = env.reset()
        done = False
        ep_return = 0.0
        trajectory = []
        
        while not done:
            obs_list = [obs[env.agents[i]] for i in range(env.n_agents)]
            actions = agent.get_actions(obs_list)
            
            action_dict = {env.agents[i]: actions[i] for i in range(env.n_agents)}
            next_obs, next_state, shared_reward, done = env.step(action_dict)
            next_obs_list = [next_obs[env.agents[i]] for i in range(env.n_agents)]
            
            trajectory.append((obs_list, state, actions, shared_reward, next_obs_list, next_state, done))
            obs, state = next_obs, next_state
            
            ep_return += shared_reward
            
        # Execute fundamental push bounds
        buffer.push(trajectory)
        ep_returns.append(ep_return)
        
        c1_loss, c2_loss = 0.0, 0.0
        # Initialize optimization structure inherently after a trivial transition warmup period sequentially
        if len(buffer) > 10: 
            batch = buffer.sample(batch_size, agent.k)
            c1_loss, c2_loss = agent.update(batch)
            
        pd.DataFrame([{
            "Episode": ep, "Reward": ep_return, "Critic1 Loss": c1_loss, "Critic2 Loss": c2_loss
        }]).to_csv(csv_file, mode='a', header=False, index=False)
        
        if ep % 100 == 0:
            avg_rtn = np.mean(ep_returns[-100:])
            print(f"Episode {ep:05d} \t Avg Return: {avg_rtn:8.2f} \t C1: {c1_loss:.4f} \t C2: {c2_loss:.4f}")

    print("Completed TCCA-SAC Off-Policy Training!")
    env.close()

if __name__ == '__main__':
    train()
