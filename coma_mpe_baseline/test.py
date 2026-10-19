import time
from env_wrapper import SimpleSpreadWrapper
from coma_agent import COMAMultiAgent

def test():
    # Adding render_mode="human" allows Pygame visualization of the MPE interface natively
    env = SimpleSpreadWrapper(N=3, max_cycles=50, render_mode="human")
    
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    state_dim = env.state_dim
    n_actions = env.n_actions
    
    agent = COMAMultiAgent(n_agents, obs_dim, state_dim, n_actions)
    
    # Load previously trained models
    try:
        agent.load_models()
        print("Loaded trained model weights successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you run `python train.py` completely to save the weights before testing!")
        return

    print("Running visualizer test...")
    
    # Standard visualization evaluation loop without optimization gradients
    for ep in range(5): # Test over 5 episodes
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        
        while not done:
            obs_list = [obs[env.agents[i]] for i in range(n_agents)]
            
            # Extract standard actions through localized actors
            actions, _ = agent.get_actions(obs_list)
            
            action_dict = {env.agents[i]: actions[i] for i in range(n_agents)}
            
            next_obs, _, shared_reward, done = env.step(action_dict)
            
            obs = next_obs
            ep_return += shared_reward
            time.sleep(0.05) # Add small frame delay for clean visualization speed to user
            
        print(f"Test Episode {ep + 1} finalized | Shared Return: {ep_return:.2f}")

    env.close()

if __name__ == '__main__':
    test()
