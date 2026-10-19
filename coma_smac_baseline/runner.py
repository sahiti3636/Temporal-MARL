import numpy as np

class Runner:
    """
    Manages structured trajectory extraction cleanly executing SMAC validation cycles natively.
    Maintains sequential actor tracking arrays natively evaluating boolean masks appropriately.
    """
    def __init__(self, env, n_agents, n_actions):
        self.env = env
        self.n_agents = n_agents
        self.n_actions = n_actions

    def run_episode(self, agent):
        self.env.reset()
        terminated = False
        episode_return = 0.0
        
        # Track initial chronological GRU state structures 
        hidden_states = agent.init_hidden()
        last_actions = np.zeros((self.n_agents,), dtype=np.int32)
        
        episode_data = []
        
        while not terminated:
            # Gather state architectures specifically configured via StarCraft dimensions
            obs = np.array(self.env.get_obs()) 
            state = self.env.get_state()      
            avail_actions = np.array(self.env.get_avail_actions()) 
            
            # Execute dynamically valid actor configuration profiles
            actions, next_hidden_states = agent.get_actions(obs, last_actions, hidden_states, avail_actions)
            
            reward, terminated, info = self.env.step(actions)
            episode_return += reward
            
            # Cache array components exclusively for training bounds
            episode_data.append({
                "obs": obs,
                "state": state,
                "actions": actions,
                "avail_actions": avail_actions,
                "reward": reward,
                "last_actions": last_actions.copy()
            })
            
            last_actions = np.array(actions)
            hidden_states = next_hidden_states
            
        win = info.get("battle_won", False)
        
        return episode_data, episode_return, win
