import numpy as np
from pettingzoo.mpe import simple_spread_v3

class SimpleSpreadWrapper:
    """
    A lightweight wrapper around simple_spread_v3 to extract global states
    and handle the shared cooperative reward. Continuous observation space,
    discrete action space.
    """
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode=None):
        # Enforcing discrete actions as required for basic COMA implementation
        self.env = simple_spread_v3.parallel_env(
            N=N, 
            local_ratio=local_ratio, 
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions,
            render_mode=render_mode
        )
        self.agents = self.env.possible_agents
        
        # Reset once to get observation and action spaces
        obs, _ = self.env.reset()
        
        self.obs_dim = obs[self.agents[0]].shape[0]
        # action space for simple_spread with discrete is Discrete(5)
        self.n_actions = self.env.action_space(self.agents[0]).n
        self.n_agents = len(self.agents)
        
        # Global state dimension is just the concatenation of all agent local observations
        self.state_dim = self.obs_dim * self.n_agents

    def reset(self):
        obs, info = self.env.reset()
        state = self._get_global_state(obs)
        return obs, state

    def step(self, actions_dict):
        obs, rewards, terminations, truncations, infos = self.env.step(actions_dict)
        state = self._get_global_state(obs)
        
        # simple_spread_v3 returns identical shared reward for all agents natively,
        # but we average to get a single scalar for our centralized training architecture.
        shared_reward = float(np.mean(list(rewards.values())))
        
        # Check if episode is done based on max_cycles (truncations)
        done = all(terminations.values()) or all(truncations.values())
        
        return obs, state, shared_reward, done

    def _get_global_state(self, obs_dict):
        """
        Concatenate all individual agent observations into a single 1D global state vector.
        Maintains consistent agent ordering.
        """
        state = np.concatenate([obs_dict[agent] for agent in self.agents])
        return state
        
    def close(self):
        self.env.close()
