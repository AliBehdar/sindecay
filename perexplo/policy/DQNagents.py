import random,os
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from utility.networks import Dueling_DQN_MLP_Network,DQN_MLP_Network
import logging

logger = logging.getLogger(__name__)
class DQNAgent:
    def __init__(self, env: gym.Env,cfg):
        
        self.cfg=cfg
        self.env = env
        self.lr=cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.target_updates = cfg.target_updates
        self.action_space = env.action_space
        self.action_space.seed(self.cfg.seed)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.device=cfg.device
        if cfg.model_name=="DQN":
            self.dqn = DQN_MLP_Network(self.state_size[0], self.action_size,cfg).to(self.device)
            self.dqn_target = DQN_MLP_Network(self.state_size[0], self.action_size,cfg).to(self.device)
        else:
            self.dqn = Dueling_DQN_MLP_Network(self.state_size[0], self.action_size, cfg).to(self.device)
            self.dqn_target = Dueling_DQN_MLP_Network(self.state_size[0], self.action_size, cfg).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=self.lr)
        self.memory = deque(maxlen=self.cfg.memory_size)
        self.tau = cfg.tau 
        self.gamma = cfg.gamma
        self.update_counter = 0  
        self.clip_grad_norm=cfg.clip_grad_norm
    def _process_state(self, state):
        if isinstance(state, dict):
            print("isinstance(state, dict")
            return np.concatenate([state[key].flatten() for key in sorted(state.keys())])
        else:
            return np.asarray(state).flatten()
        
    def select_action_g(self, state, epsilon):

        state = self._process_state(state)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device=self.device)
       
        with torch.no_grad():
            q_value = self.dqn(state)[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = torch.argmax(q_value).item() 
        return action
    
    def select_action_b(self, state,temperature):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        Q_values = self.dqn(state)
        action_probs = F.softmax(Q_values / temperature, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def append_sample(self, state, action, reward, next_state, done):
      
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def train_DoubleDQN(self):

        batch = random.sample(self.memory, self.batch_size)

        states      = np.array([i[0] for i in batch])
        actions     = np.array([i[1] for i in batch])
        rewards     = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones       = np.array([i[4] for i in batch])

        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)

        next_Qs = self.dqn(next_states)
        next_action = torch.argmax(next_Qs, dim=1)

        with torch.no_grad():
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
    
    def train_DQN(self):
        batch = random.sample(self.memory, self.batch_size)

        states      = np.array([i[0] for i in batch])
        actions     = np.array([i[1] for i in batch])
        rewards     = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones       = np.array([i[4] for i in batch])

        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)


        with torch.no_grad():
            next_Q_target = self.dqn_target(next_states)        
            max_next_Q, _ = next_Q_target.max(dim=1)           
            mask = 1.0 - dones
            target_value = rewards + self.gamma * max_next_Q * mask  

        curr_Qs = self.dqn(states)                     
        main_value = curr_Qs.gather(1, actions.unsqueeze(1)).squeeze(1)  

        # loss (squared error / 2 to match your original)
        error = ((main_value - target_value) ** 2) / 2.0
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

        if self.cfg.myAl:
            y = np.exp(-self.cfg.decay_rate * episode ) * (1 + self.cfg.cos_amp * np.cos(self.cfg.cos_freq * episode + self.cfg.cos_phase))
            epsilon=self.cfg.min_epsilon + (self.cfg.max_epsilon - self.cfg.min_epsilon)* y
        else:
            y =  np.exp(-self.cfg.decay_rate*episode) # 
            epsilon=self.cfg.min_epsilon + (self.cfg.max_epsilon - self.cfg.min_epsilon)* y
      
        return epsilon 
    def update_tempertuer(self,episode):
   
        if self.cfg.myAl:
            y = np.exp(-self.cfg.periodic_decay * episode ) * (1 + self.cfg.cos_amp * np.cos(self.cfg.cos_freq * episode + self.cfg.cos_phase))
            tempertuer =self.cfg.min_tempertuer + (self.cfg.max_tempertuer - self.cfg.min_tempertuer)* y
        else:
            y =  np.exp(-self.cfg.decay_rate*episode) # 
            tempertuer =self.cfg.min_tempertuer + (self.cfg.max_tempertuer - self.cfg.min_tempertuer)* y
        return tempertuer  
    
    def load(self, path):
        if self.cfg.model_name=="DQN": 
            self.dqn =DQN_MLP_Network(self.state_size[0], self.action_size, self.cfg).to(self.device)   
        else:
            self.dqn =Dueling_DQN_MLP_Network(self.state_size[0], self.action_size, self.cfg).to(self.device)   
        self.dqn.load_state_dict(torch.load(path,map_location=self.device))
        self.dqn_target.load_state_dict(torch.load(path,map_location=self.device))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.dqn.state_dict(), path)

