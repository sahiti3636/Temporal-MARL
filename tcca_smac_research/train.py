import os
import csv
import torch
import numpy as np
from smac.env import StarCraft2Env

from tcca_agent import TCCAAgent
from replay_buffer import ReplayBuffer
from runner import Runner

class Args:
    # Environment
    map_name = "3m"
    max_ep_len = 100 # Default max length
    
    # Network
    rnn_hidden_dim = 64
    critic_hidden_dim = 64
    
    # Training
    lr = 5e-4
    gamma = 0.99
    alpha = 0.05 # Entropy coefficient
    tau = 0.005 # Soft update rate
    grad_norm_clip = 10.0
    
    # TCCA
    k_steps = 3
    
    # Execution
    cuda = torch.cuda.is_available()
    batch_size = 32
    buffer_capacity = 5000
    episodes = 50000
    train_interval = 1 # Train every N episodes
    updates_per_train = 4
    
    log_interval = 20
    
def main():
    args = Args()
    
    env = StarCraft2Env(map_name=args.map_name)
    env_info = env.get_env_info()
    
    args.num_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.max_ep_len = env_info.get("episode_limit", 100)
    
    args.actor_input_dim = args.obs_shape + args.n_actions + args.num_agents
    args.critic_input_dim = args.state_shape + args.num_agents * args.n_actions + args.num_agents
    
    print("Environment Info:", env_info)
    print("Using CUDA:", args.cuda)
    
    buffer = ReplayBuffer(args.buffer_capacity, args.max_ep_len, args.num_agents, args.state_shape, args.actor_input_dim, args.n_actions, args.rnn_hidden_dim)
    agent = TCCAAgent(args)
    runner = Runner(env, agent, buffer, args)
    
    # Logging setup
    log_file = "tcca_logs.csv"
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Average Reward", "Win Rate", "Critic MSE", "Actor Loss"])
        
    reward_history = []
    win_history = []
    critic_loss_history = []
    actor_loss_history = []
    
    for ep in range(1, args.episodes + 1):
        ep_reward, win, steps = runner.run_episode(test_mode=False)
        reward_history.append(ep_reward)
        win_history.append(1.0 if win else 0.0)
        
        # Train
        if buffer.current_size >= args.batch_size and ep % args.train_interval == 0:
            for _ in range(args.updates_per_train):
                batch = buffer.sample_k_step_segments(args.batch_size, k=args.k_steps)
                actor_loss, critic_loss = agent.update(batch)
                
                actor_loss_history.append(actor_loss)
                critic_loss_history.append(critic_loss)
                
        if ep % args.log_interval == 0:
            avg_reward = np.mean(reward_history[-args.log_interval:])
            win_rate = np.mean(win_history[-args.log_interval:])
            avg_critic_loss = np.mean(critic_loss_history[-args.log_interval:]) if critic_loss_history else 0
            avg_actor_loss = np.mean(actor_loss_history[-args.log_interval:]) if actor_loss_history else 0
            
            print(f"Episode: {ep}/{args.episodes} | Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Critic MSE: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
            
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([ep, avg_reward, win_rate, avg_critic_loss, avg_actor_loss])
                
    env.close()
    print("Training finished. Logs saved to", log_file)

if __name__ == "__main__":
    main()
