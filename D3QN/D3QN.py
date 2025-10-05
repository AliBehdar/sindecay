import pygame, random
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import hydra 
from omegaconf import DictConfig
from collections import deque
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add the parent directory (containing 'utility') to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utility.plots import plot
from utility.utility_off import seed,NetworkCNN,NetworkMLP
# 1.1
import gymnasium as gym
import ale_py

class DQNAgent:
    def __init__(self, env: gym.Env,cfg):
        
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        self.device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg=cfg
        self.env = env
        self.lr=cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.target_updates = cfg.target_updates
        self.action_space = env.action_space

        self.action_space.seed(self.cfg.seed)
        #self.state_size = self.env.observation_space.shape
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n


        self.is_ram = (cfg.obs_type == "ram")

        if self.is_ram:
            self.dqn = NetworkMLP(self.state_size[0], self.action_size, cfg).to(self.device)
            self.dqn_target = NetworkMLP(self.state_size[0], self.action_size, cfg).to(self.device)
        else:
            self.dqn = NetworkCNN(self.state_size, self.action_size, cfg).to(self.device)
            self.dqn_target = NetworkCNN(self.state_size, self.action_size, cfg).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=self.lr)
        self.memory = deque(maxlen=self.cfg.memory_size)
        
        self.Soft_Update = False 
        self.tau = cfg.tau 
        self.loss_history=[]
        self.gamma = cfg.gamma
        self.update_counter = 0  
        self.soft_update = cfg.soft_update 
     

    def get_action(self, state, epsilon):
        #state = self._process_state(state)
        state_np = np.array(state)
        state = torch.from_numpy(state_np).float().to(self.device)
        if state.dim() == 2:  # Single grayscale (h, w)
            state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        elif state.dim() == 1:  # RAM (features,)
            state = state.unsqueeze(0)  # (1, features)
        elif state.dim() == 3 :  # Single RGB (h, w, c)
            if self.cfg.obs_type == "rgb_image":
                state = state.unsqueeze(0).permute(0, 3, 1, 2)  # For RGB: (1, c, h, w)
            else:
                state = state.unsqueeze(0)  # (1, c, h, w)     
       
        with torch.no_grad():
            q_value = self.dqn(state)[0]

        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = torch.argmax(q_value).item() 
        return action
   
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
 
    def train_step(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states      = np.array([i[0] for i in mini_batch])
        actions     = np.array([i[1] for i in mini_batch])
        rewards     = np.array([i[2] for i in mini_batch])
        next_states = np.array([i[3] for i in mini_batch])
        dones       = np.array([i[4] for i in mini_batch])

        states      = torch.from_numpy(states).float().to(self.device)
        if not self.is_ram:
            if len(states.shape) == 3:  # (batch, h, w) for grayscale
                states = states.unsqueeze(1)  # (batch, 1, h, w)
            #else:  # (batch, h, w, c) for RGB
            #    states = states.permute(0, 3, 1, 2)  # (batch, c, h, w)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device) 
        if not self.is_ram:
            if len(next_states.shape) == 3:  # (batch, h, w) for grayscale
                next_states = next_states.unsqueeze(1)  # (batch, 1, h, w)
            #else:  # (batch, h, w, c) for RGB
            #    next_states = next_states.permute(0, 3, 1, 2)  # (batch, c, h, w)
        dones       = torch.from_numpy(dones).float().to(self.device)

        next_Qs = self.dqn(next_states)
        next_action = torch.argmax(next_Qs, dim=1)
        next_Q_targs = self.dqn_target(next_states)
        target_value = next_Q_targs.gather(1, next_action.unsqueeze(1)).squeeze(1)

        mask = 1 - dones
        target_value = rewards + self.gamma * target_value * mask
        
        curr_Qs = self.dqn(states)
        main_value = curr_Qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        error = (main_value - target_value) ** 2 * 0.5
        loss = torch.mean(error)
        self.loss_history.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def _target_update(self):
        if self.soft_update:
            # Soft update: Do this frequently (e.g., every train_step)
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
            # Hard update: Do this periodically (use counter)
            self.update_counter += 1
            if self.update_counter % self.target_updates == 0:
                self._target_hard_update()
        
    def update_Gamma(self):
        self.gamma = 1 - 0.985 * (1 - self.gamma)

    def load(self, path):

        if self.is_ram:
            self.dqn = NetworkMLP(self.state_size[0], self.action_size, self.cfg).to(self.device)
        else:
            self.dqn = NetworkCNN(self.state_size, self.action_size, self.cfg).to(self.device)
            
        self.dqn.load_state_dict(torch.load(path,map_location=self.device))
        self._target_hard_update()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.dqn.state_dict(), path)


@hydra.main(version_base="1.1", config_path="./conf", config_name="configs")
def main(cfg: DictConfig):
    seed(cfg)
    gym.register_envs(ale_py) 
    env = gym.make(cfg.env_name,render_mode="human" if not cfg.train else None,obs_type=cfg.obs_type)
    env=gym.wrappers.ResizeObservation(env, (84, 84)) 
    env = gym.wrappers.FrameStackObservation(env, 4)
    agent = DQNAgent(env,cfg)
    if cfg.train:
        update_cnt = 0
        reward_history=[]
        epsilon_history=[] 
        epsilon=cfg.epsilon  
        for episode in range(1,cfg.max_episodes+1):
            state = agent.env.reset()
            state=state[0]
            episode_reward = 0
            done = False  
            while not done :
                update_cnt += 1
                action = agent.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _= agent.env.step(action)
                if isinstance(state, tuple): 
                    next_state = next_state[0]
                agent.append_sample(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
                # if episode ends
                if done:
                    print("Episode: {}/{}, Episodes reward: {:.6}, e: {:.3}".format(episode, cfg.max_episodes, episode_reward, epsilon)) 
                    break

                if update_cnt >= agent.batch_size and update_cnt % 4 == 0:
                    agent.train_step()
                    if cfg.soft_update:
                        agent._target_update()
                    elif update_cnt % agent.target_updates == 0:
                        agent._target_hard_update()

            reward_history.append(episode_reward)   
            epsilon_history.append(epsilon)
           
            if cfg.myAl:
                y = np.exp(-0.01 * episode) * (1 + 0.5 * np.cos(0.2 * episode + 0.5))
                #y = np.exp(-cfg.decay_rate * x) * (1 + cfg.cos_amp * np.cos(cfg.cos_freq * x + cfg.cos_phase))
            else:
                y =  np.exp(-cfg.decay_rate*episode)
            
            epsilon = cfg.min_epsilon + (cfg.max_epsilon - cfg.min_epsilon)* y
            
            if episode % cfg.save_interval==0:
                agent.save(cfg.save_path + '_' + f'{episode}')
                plot(episode,cfg,reward_history,epsilon_history,agent.loss_history)

    else:
        agent.load(cfg.load_path)
        
        for episode in range(5):
            state = agent.env.reset(cfg.seed)
            state=state[0]
            episode_reward = 0
            done = False  
            while not done:
                action = agent.get_action(state,0.01)
                next_state, reward, terminated, truncated, _= agent.env.step(action)
                if isinstance(state, tuple): 
                    next_state = next_state[0]
                done = terminated or truncated
                agent.append_sample(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
                if done: 
                    print("Episode: {}/{}, Episodes reward: {:.4}, e: {}".format(episode+1, cfg.max_episodes, episode_reward, 0.01)) 
                    break
        pygame.quit()

if __name__=="__main__":
    main()