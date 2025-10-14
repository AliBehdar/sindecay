import pygame,random
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import hydra 
from omegaconf import DictConfig
from collections import deque
import os
#from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from plots import plot
from utility_off import seed,NetworkMLP

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
        #self.device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg=cfg
        self.env = env
        self.lr=cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.target_updates = cfg.target_updates
        self.action_space = env.action_space

        self.action_space.seed(self.cfg.seed)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n


        self.dqn = NetworkMLP(self.state_size[0], self.action_size, cfg)
        self.dqn_target = NetworkMLP(self.state_size[0], self.action_size, cfg)
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=self.lr)
        self.memory = deque(maxlen=self.cfg.memory_size)
        self.tau = cfg.tau 
        self.gamma = cfg.gamma
        self.update_counter = 0  

    def _process_state(self, state):
        if isinstance(state, dict):
            print("isinstance(state, dict")
            return np.concatenate([state[key].flatten() for key in sorted(state.keys())])
        else:
            return np.asarray(state).flatten()
        
    def get_action(self, state, epsilon):

        state = self._process_state(state)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)  
       
        with torch.no_grad():
            q_value = self.dqn(state)[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = torch.argmax(q_value).item() 
        return action
   
    def append_sample(self, state, action, reward, next_state, done):
      
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):

        batch = random.sample(self.memory, self.batch_size)

        states      = np.array([i[0] for i in batch])
        actions     = np.array([i[1] for i in batch])
        rewards     = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones       = np.array([i[4] for i in batch])

        states      = torch.from_numpy(states).float()
        actions     = torch.from_numpy(actions).long()
        rewards     = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones       = torch.from_numpy(dones).float()

        next_Qs = self.dqn(next_states)
        next_action = torch.argmax(next_Qs, dim=1)
        next_Q_targs = self.dqn_target(next_states)
        target_value = next_Q_targs.gather(1, next_action.unsqueeze(1)).squeeze(1)

        mask = 1 - dones
        target_value = rewards + self.gamma * target_value * mask
        
        curr_Qs = self.dqn(states)
    
        main_value = curr_Qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        error = ((main_value - target_value) ** 2 / 2.0) 
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _target_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def update_Gamma(self):
        self.gamma = 1 - 0.985 * (1 - self.gamma)
    
    def update_epsilon(self,episode):
         
        if episode <= self.cfg.max_episodes - (self.cfg.max_episodes /5):           
            if self.cfg.myAl:
                y = np.exp(-self.cfg.decay_rate * episode ) * (1 + self.cfg.cos_amp * np.cos(self.cfg.cos_freq * episode + self.cfg.cos_phase))
                epsilon=self.cfg.min_epsilon + (self.cfg.max_epsilon - self.cfg.min_epsilon)* y
            else:
                y =  np.exp(-self.cfg.decay_rate1*episode) # 
                epsilon=self.cfg.min_epsilon + (self.cfg.max_epsilon - self.cfg.min_epsilon)* y
        else:
            epsilon=0.01
            
        return epsilon 

    def load(self, path):

        self.dqn = NetworkMLP(self.state_size[0], self.action_size, self.cfg).to(self.device)   
        self.dqn.load_state_dict(torch.load(path,map_location=self.device))
        self.dqn_target.load_state_dict(torch.load(path,map_location=self.device))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.dqn.state_dict(), path)


@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    seed(cfg)
    env = gym.make(cfg.env_name,render_mode="human" if not cfg.train else None,
                   continuous=False, gravity=-9.8,enable_wind=False,
                     wind_power=10.0, turbulence_power=1.5)
    agent = DQNAgent(env,cfg)
    if cfg.train:
        update_cnt = 0
        reward_history=[]
        epsilon_history=[] 
        loss_history =[]
        epsilon=cfg.epsilon  
        for episode in range(1,cfg.max_episodes+1):
            state = agent.env.reset(seed=cfg.seed+episode)
            state=state[0]
            episode_reward = 0
            episode_loss=0
            done = False  
            while not done :
                update_cnt += 1
                action = agent.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _= agent.env.step(action)
                if isinstance(state, tuple): 
                    next_state = next_state[0]
                done = terminated or truncated
                agent.append_sample(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
                # if episode ends
                if done:
                    print("Episode: {}/{}, Episodes reward: {:.6}, e: {:.3}".format(episode, cfg.max_episodes, episode_reward, epsilon)) 
                    break
                if update_cnt >= agent.batch_size:
                    loss=agent.train_step()
                    agent.update_Gamma()
                    episode_loss +=loss

                    if update_cnt % agent.target_updates == 0:
                        agent._target_update()
            
            loss_history.append(episode_loss)
            reward_history.append(episode_reward)   
            epsilon_history.append(epsilon)
            
            epsilon=agent.update_epsilon(episode)
            
            if episode % cfg.save_interval==0:
                agent.save(cfg.save_path + '_' + f'{episode}')
                plot(episode,cfg,reward_history,epsilon_history,loss_history)

    elif cfg.test:
        agent.load(cfg.load_path)
        sum_reward=[]
        
        for episode in range(100):
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
                state = next_state
                episode_reward += reward
                if done: 
                    print("Episode: {}/{}, Episodes reward: {:.4}, e: {}".format(episode+1, cfg.max_episodes, episode_reward, 0.01)) 
                    break
            sum_reward.append(episode_reward)
            
        print(sum(sum_reward))

        pygame.quit()

if __name__=="__main__":
    main()