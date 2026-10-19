import numpy as np
from env_wrapper import SimpleSpreadWrapper
from coma_agent import COMAMultiAgent

def train():
    # Setup simple_spread_v3 environment instance via our wrapper
    env = SimpleSpreadWrapper(N=3, max_cycles=25)
    
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    state_dim = env.state_dim
    n_actions = env.n_actions
    
    # Initialize Multi-Agent COMA Algorithm
    agent = COMAMultiAgent(n_agents, obs_dim, state_dim, n_actions)
    
    num_episodes = 50000
    batch_size = 32 # train/update after collecting `batch_size` full episodes
    
    print(f"Starting training wrapper on simple_spread_v3...")
    print(f"Agents: {n_agents} | Obs Dim: {obs_dim} | State Dim (Global): {state_dim} | Actions: {n_actions}")
    
    ep_returns = []
    
    rollouts = []
    critic_loss = 0.0 # for initialized printing
    
    for ep in range(1, num_episodes + 1):
        obs, state = env.reset()
        done = False
        ep_return = 0.0
        
        # Rollout Phase
        while not done:
            # We strictly listify local observations over agents based on defined environment agent order
            obs_list = [obs[env.agents[i]] for i in range(n_agents)]
            
            actions, _ = agent.get_actions(obs_list)
            
            action_dict = {env.agents[i]: actions[i] for i in range(n_agents)}
            
            next_obs, next_state, shared_reward, done = env.step(action_dict)
            
            next_obs_list = [next_obs[env.agents[i]] for i in range(n_agents)]
            
            # Compile SARS transition
            rollouts.append((obs_list, state, actions, shared_reward, next_obs_list, next_state, done))
            
            obs = next_obs
            state = next_state
            ep_return += shared_reward
            
        ep_returns.append(ep_return)
        
        # Update Phase
        if ep % batch_size == 0:
            critic_loss = agent.update(rollouts)
            rollouts = [] # clear memory after update
            
        # Logging Phase
        if ep % 100 == 0:
            avg_return = np.mean(ep_returns[-100:])
            # Printed critic loss serves as explicitly tracked Bellman Error metric
            print(f"Episode {ep:05d} \t Avg Shared Return (last 100): {avg_return:8.2f} \t Critic Loss (Bellman): {critic_loss:.4f}")

    print("Completed Training!")
    agent.save_models()
    print("Models saved locally.")
    env.close()

if __name__ == '__main__':
    train()
