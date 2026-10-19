import pandas as pd
import numpy as np
import os
from smac.env import StarCraft2Env
from runner import Runner
from coma_agent import COMAMACAgent

def train():
    # Instantiate StarCraft component handling '3m' environment bounds successfully
    env = StarCraft2Env(map_name="3m")
    env_info = env.get_env_info()
    
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    
    agent = COMAMACAgent(n_agents, obs_dim, state_dim, n_actions)
    runner = Runner(env, n_agents, n_actions)
    
    num_episodes = 50000
    batch_size = 8
    
    csv_file = "training_logs.csv"
    if os.path.exists(csv_file): 
        os.remove(csv_file)
        
    pd.DataFrame(columns=["Episode", "Win Rate", "Total Reward", "Average Critic Loss"]).to_csv(csv_file, index=False)
    
    ep_returns = []
    ep_wins = []
    batch_episodes = []
    
    print(f"Executing Sequential SMAC GRU Training on Configuration: {n_agents} Marines...")
    
    for ep in range(1, num_episodes + 1):
        ep_data, ep_return, win = runner.run_episode(agent)
        
        batch_episodes.append(ep_data)
        ep_returns.append(ep_return)
        ep_wins.append(1 if win else 0)
        
        c_loss = 0.0
        if len(batch_episodes) >= batch_size:
            c_loss = agent.update(batch_episodes)
            batch_episodes = []
            
        if ep % 100 == 0:
            win_rate = np.mean(ep_wins[-100:])
            avg_return = np.mean(ep_returns[-100:])
            print(f"Ep {ep} \t Win Rate: {win_rate:.2f} \t Reward: {avg_return:.2f} \t C_Loss: {c_loss:.4f}")
            
            # Export dynamically to ensure validation scripts function consistently downstream
            pd.DataFrame([{
                "Episode": ep, "Win Rate": win_rate, "Total Reward": avg_return, "Average Critic Loss": c_loss
            }]).to_csv(csv_file, mode='a', header=False, index=False)

    print("Completed Execution of Training Sequences!")
    env.close()

if __name__ == '__main__':
    train()
