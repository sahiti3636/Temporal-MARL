import numpy as np
import torch

class Runner:
    def __init__(self, env, agent, replay_buffer, args):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.args = args
        self.num_agents = args.num_agents
        self.n_actions = args.n_actions
        
    def get_actor_inputs(self, obs, last_action_onehot):
        # obs: (num_agents, obs_shape)
        # last_action_onehot: (num_agents, n_actions)
        
        # Build agent_id onehot
        agent_id_onehot = np.eye(self.num_agents, dtype=np.float32)
        
        inputs = np.concatenate([obs, last_action_onehot, agent_id_onehot], axis=1)
        return inputs

    def run_episode(self, test_mode=False):
        self.env.reset()
        terminated = False
        step = 0
        
        episode_reward = 0.0
        
        # Initialize hidden state
        hidden_state = self.agent.actor.init_hidden().expand(self.num_agents, -1).contiguous()
        last_action_onehot = np.zeros((self.num_agents, self.n_actions), dtype=np.float32)
        
        episode_data = {
            "obs": [],
            "state": [],
            "actions": [],
            "actions_onehot": [],
            "rewards": [],
            "dones": [],
            "avail_actions": [],
            "hidden_states": []
        }
        
        while not terminated and step < self.args.max_ep_len:
            obs = np.array(self.env.get_obs(), dtype=np.float32)
            state = np.array(self.env.get_state(), dtype=np.float32)
            avail_actions = np.array(self.env.get_avail_actions(), dtype=np.float32)
            
            actor_inputs = self.get_actor_inputs(obs, last_action_onehot)
            
            # Record before action
            episode_data["obs"].append(actor_inputs)
            episode_data["state"].append(state)
            episode_data["avail_actions"].append(avail_actions)
            episode_data["hidden_states"].append(hidden_state.detach().cpu().numpy())
            
            # Select actions
            
            inputs_t = torch.tensor(actor_inputs, dtype=torch.float32)
            avail_actions_t = torch.tensor(avail_actions, dtype=torch.float32)
            if self.args.cuda:
                inputs_t = inputs_t.cuda()
                avail_actions_t = avail_actions_t.cuda()
            
            with torch.no_grad():
                actions_t, next_hidden = self.agent.select_actions(inputs_t, hidden_state, avail_actions_t, test_mode=test_mode)
            
            hidden_state = next_hidden
            actions = actions_t.cpu().numpy()
            
            # Step in environment
            reward, terminated, info = self.env.step(actions.tolist())
            
            # Store actions
            actions_expanded = np.expand_dims(actions, axis=-1)
            episode_data["actions"].append(actions_expanded)
            
            actions_onehot = np.zeros((self.num_agents, self.n_actions), dtype=np.float32)
            actions_onehot[np.arange(self.num_agents), actions] = 1.0
            episode_data["actions_onehot"].append(actions_onehot)
            
            episode_data["rewards"].append(np.array([reward], dtype=np.float32))
            episode_data["dones"].append(np.array([float(terminated)], dtype=np.float32))
            
            last_action_onehot = actions_onehot
            episode_reward += reward
            step += 1
            
        # Append terminal state
        obs = np.array(self.env.get_obs(), dtype=np.float32)
        state = np.array(self.env.get_state(), dtype=np.float32)
        avail_actions = np.array(self.env.get_avail_actions(), dtype=np.float32)
        actor_inputs = self.get_actor_inputs(obs, last_action_onehot)
        
        episode_data["obs"].append(actor_inputs)
        episode_data["state"].append(state)
        episode_data["avail_actions"].append(avail_actions)
        episode_data["hidden_states"].append(hidden_state.detach().cpu().numpy())
        
        # We need actions and onehot for terminal state just to match shapes in buffer
        # (they won't be used since step k is done)
        dummy_actions = np.zeros((self.num_agents, 1), dtype=np.int64)
        dummy_onehot = np.zeros((self.num_agents, self.n_actions), dtype=np.float32)
        episode_data["actions"].append(dummy_actions)
        episode_data["actions_onehot"].append(dummy_onehot)
        
        # Convert lists to numpy arrays
        for k in episode_data.keys():
            episode_data[k] = np.array(episode_data[k])
            
        episode_data["ep_len"] = step
        
        win = info.get("battle_won", False)
        
        if not test_mode:
            self.replay_buffer.insert_episode(episode_data)
            
        return episode_reward, win, step
