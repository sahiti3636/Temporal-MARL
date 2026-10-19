import numpy as np
import pandas as pd
import os
from env_wrapper import SimpleSpreadWrapper
from tcca_agent import TCCAAgent

def train():
    env = SimpleSpreadWrapper(N=3, max_cycles=25)
    
    agent = TCCAAgent(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, k=5)
    
    num_episodes = 50000
    batch_size = 32
    
    csv_file = "training_logs.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
        
    print(f"Starting TCCA training run... Outputting to {csv_file}")
    
    df = pd.DataFrame(columns=["Episode", "Total Reward", "Average Critic Loss"])
    df.to_csv(csv_file, index=False)
    
    ep_returns = []
    trajectories = []
    last_critic_loss = 0.0
    
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
            
            trajectory.append((obs_list, state, actions, shared_reward, next_obs, next_state, done))
            
            obs = next_obs
            state = next_state
            ep_return += shared_reward
            
        trajectories.append(trajectory)
        ep_returns.append(ep_return)
        
        if len(trajectories) >= batch_size:
            last_critic_loss = agent.update(trajectories)
            trajectories = [] # clear memory
            
        # Logging sequence structured exclusively via pandas per your parameters
        row = pd.DataFrame([{
            "Episode": ep,
            "Total Reward": ep_return,
            "Average Critic Loss": last_critic_loss
        }])
        row.to_csv(csv_file, mode='a', header=False, index=False)
        
        if ep % 100 == 0:
            avg_return = np.mean(ep_returns[-100:])
            print(f"Episode {ep:05d} \t Avg Return: {avg_return:8.2f} \t Critic Loss: {last_critic_loss:.4f}")

    print("Completed Training!")
    env.close()

if __name__ == '__main__':
    train()
