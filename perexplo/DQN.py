
import torch
import pygame,os
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from utility_off import ReplayMemory,seed,DQN_Network
import torch.nn.functional as F
import hydra 
from omegaconf import DictConfig
from plots import plot
import logging

logger = logging.getLogger(__name__)


class DQN_Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """
    
    def __init__(self, env,device,clip_grad_norm, learning_rate,
                 discount,memory_capacity,seed,tau,discount_factor):

        self.loss_history = []
        self.learned_counts = 0
        self.temperature_history=[]
        self.epsilon_history=[]
        self.discount      = discount
        self.device=device 
        self.action_space  = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity,device)
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.shape[0]).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.shape[0]).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.clip_grad_norm = clip_grad_norm 
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.tau=tau       
        self.discount_factor=discount_factor
    def select_action_g(self, state,epsilon):

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if np.random.random() < epsilon:
            return self.action_space.sample()

        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
                        
            return action


    def select_action_b(self, state,temperature):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        Q_values = self.main_network(state)
        action_probs = F.softmax(Q_values / temperature, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def learn(self, batch_size, done):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions       = actions.unsqueeze(1)
        rewards       = rewards.unsqueeze(1)
        dones         = dones.unsqueeze(1)       
        predicted_q = self.main_network(states)
        predicted_q = predicted_q.gather(dim=1, index=actions) 
        with torch.no_grad():            
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0] 
        next_target_q_value[dones] = 0 
        y_js = rewards + (self.discount * next_target_q_value) 
        loss = self.critertion(predicted_q, y_js)  
        self.optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step() 
        return loss.item()

    def _target_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.main_network.state_dict(), path)
    
    def update_Gamma(self):
        self.discount_factor = 1 - 0.98 * (1 - self.discount_factor)
       
    
class Model_TrainTest:
    def __init__(self, cfg):
    
        self.epsilon_greedy         = cfg.epsilon_greedy
        self.RL_load_path           = cfg.load_path
        self.save_path              = cfg.save_path
        self.save_interval          = cfg.save_interval
        self.epsilon                =cfg.epsilon_max
        self.batch_size             = cfg.batch_size
        self.update_frequency       = cfg.update_frequency
        self.max_episodes           = cfg.max_episodes
        self.max_steps              = cfg.max_steps
        self.render                 = cfg.render      
        self.render_fps             = cfg.render_fps
        self.incrising_Ep_Or_temp   = cfg.indricing_Ep_Or_temp  
        self.tempertuer             =cfg.tempertuer_max 
        self.seed=cfg.seed    
        self.cfg=cfg  
         
        # Define Env
        self.learn_frequency=cfg.learn_frequency
        self.env = gym.make(cfg.env_name, max_episode_steps=cfg.max_steps, 
                            render_mode="human" if not cfg.train else None,
                   continuous= cfg.continuous, gravity=cfg.gravity,enable_wind=cfg.enable_wind,
                     wind_power=cfg.wind_power, turbulence_power=cfg.turbulence_power)
        
        self.env.metadata['render_fps'] = self.render_fps # For max frame rate make it 0
        
        # Define the agent class
        self.agent = DQN_Agent( env               = self.env, 
                                clip_grad_norm    = cfg.clip_grad_norm,
                                learning_rate     = cfg.learning_rate,
                                discount          = cfg.discount_factor,
                                memory_capacity   = cfg.memory_capacity,
                                device            = cfg.device,
                                seed              = cfg.seed,
                                tau               =cfg.tau,
                                discount_factor   =cfg.discount_factor)
        
    def update_epsilon(self,episode):
        if self.cfg.myAl:
            y = np.exp(-self.cfg.periodic_decay * episode ) * (1 + self.cfg.cos_amp * np.cos(self.cfg.cos_freq * episode + self.cfg.cos_phase))
            epsilon=self.cfg.min_epsilon + (self.cfg.epsilon_max - self.cfg.min_epsilon)* y
        else:
            y =  np.exp(-self.cfg.epsilon_decay*episode) # 
            epsilon=self.cfg.min_epsilon + (self.cfg.epsilon_max - self.cfg.min_epsilon)* y    
        return epsilon  
    
    def update_tempertuer(self,episode):
   
        if self.cfg.myAl:
            y = np.exp(-self.cfg.periodic_decay * episode ) * (1 + self.cfg.cos_amp * np.cos(self.cfg.cos_freq * episode + self.cfg.cos_phase))
            tempertuer =self.cfg.min_tempertuer + (self.cfg.max_tempertuer - self.cfg.min_tempertuer)* y
        else:
            y =  np.exp(-self.cfg.epsilon_decay*episode) # 
            tempertuer =self.cfg.min_tempertuer + (self.cfg.max_tempertuer - self.cfg.min_tempertuer)* y
        return tempertuer   
    
    def train(self): 
        """                
        DQN training loop.
        """
        total_steps = 0
        reward_history = []
        loss_history=[] 
        epsilon_history=[]
        tempertuer_history=[]

        for episode in range(1, self.max_episodes+1):
            state = self.env.reset(seed=self.seed+episode)
            done = False      
            step_size = 0
            episode_reward = 0
            episode_loss =0                               
            while not done :
                if self.epsilon_greedy==True:
                    action = self.agent.select_action_g(state,self.epsilon)
                else:
                    action = self.agent.select_action_b(state,self.tempertuer)
                    
                next_state, reward, terminated , truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.replay_memory.store(state, action, next_state, reward, done) 

                if len(self.agent.replay_memory) > self.batch_size :
                    loss=self.agent.learn(self.batch_size, done )
                    episode_loss +=loss
                    if total_steps % self.update_frequency == 0:
                        self.agent._target_update()
                
                state = next_state
                episode_reward += reward
                step_size +=1
                # if episode ends
                if done:
                    if self.epsilon_greedy:
                        logger.info("Episode: {}/{}, Episodes reward: {:.6}, e: {:.3}".format(episode, self.cfg.max_episodes, episode_reward, self.epsilon))
                    else:
                        logger.info("Episode: {}/{}, Episodes reward: {:.6}, t: {:.3}".format(episode, self.cfg.max_episodes, episode_reward, self.tempertuer))    
                    break
                            
            loss_history.append(episode_loss)  
            epsilon_history.append(self.epsilon)
            tempertuer_history.append(self.tempertuer)
            reward_history.append(episode_reward)                       
            total_steps += step_size
            self.agent.update_Gamma()                                                              
            if self.incrising_Ep_Or_temp:
                if self.epsilon_greedy:
                    self.epsilon=self.update_epsilon(episode)
                    self.agent.epsilon_history.append(self.epsilon)
                else:
                    self.tempertuer =self.update_tempertuer(episode)
                    self.agent.epsilon_history.append(self.tempertuer)

            if episode % self.save_interval == 0:
                self.agent.save(self.cfg.save_path + '_' + f'{episode}')
                if episode != self.max_episodes:
                    plot(episode, self.cfg, reward_history, self.agent.epsilon_history, self.agent.loss_history)
                print('\n~~~~~~Interval Save: Model saved.\n')
                                                                    

    def test(self, max_episodes):  
        """                
        Reinforcement learning policy evaluation.
        """
           
        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()
        
        # Testing loop over episodes
        for episode in range(1, max_episodes+1):         
            state, _ = self.env.reset(seed=self.seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
                                                           
            while not done and not truncation:
                
                if self.epsilon_greedy:
                    action = self.agent.select_action_g(state)
                else:
                    action = self.agent.select_action_b(state)
                    
                next_state, reward, done, truncation, _ = self.env.step(action)
                                
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print log            
        result = (f"Episode: {episode}, "
                    f"Steps: {step_size:}, "
                    f"Reward: {episode_reward:.2f}, ")
        print(result)
            
        pygame.quit() # close the rendering window
        

   
@hydra.main(version_base="1.1", config_path="./conf", config_name="config_DQN")      
def main(cfg: DictConfig):
    seed(cfg)  
    DRL = Model_TrainTest(cfg) 

    if cfg.train:
        DRL.train()
    else:
        DRL.test(max_episodes = cfg.max_episodes)
if __name__ == '__main__':   
    
    main()