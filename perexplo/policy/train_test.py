 
import torch
import gymnasium as gym
from utility.plots import plot
import logging
from policy.DQNagents import DQNAgent
import pygame
import numpy as np
logger = logging.getLogger(__name__)  

class Model_TrainTest:
    def __init__(self, cfg):
    
        self.epsilon_greedy         = cfg.epsilon_greedy
        self.RL_load_path           = cfg.load_path
        self.save_path              = cfg.save_path
        self.save_interval          = cfg.save_interval
        self.batch_size             = cfg.batch_size
        self.max_episodes           = cfg.max_episodes
        self.max_steps              = cfg.max_steps
        self.tempertuer             = cfg.max_tempertuer
        self.epsilon                = cfg.max_epsilon
        self.seed=cfg.seed    
        self.cfg=cfg  
        self.target_updates = cfg.target_updates

        
        if cfg.env_name=="LunarLander-v3":
            self.env = gym.make(cfg.env_name,render_mode="human" if not cfg.train else None,
                    continuous= cfg.continuous, gravity=cfg.gravity,enable_wind=cfg.enable_wind,
                        wind_power=cfg.wind_power, turbulence_power=cfg.turbulence_power)
        elif cfg.env_name=="CartPole-v1":
            self.env = gym.make(cfg.env_name)

        self.agent = DQNAgent(self.env,cfg)
    
    
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
            state=state[0]
            episode_reward = 0
            episode_loss =0  
            done = False 

            while not done :
                total_steps +=1
                if self.epsilon_greedy==True:
                    action = self.agent.select_action_g(state,self.epsilon)
                else:
                    action = self.agent.select_action_b(state,self.tempertuer)
                    
                next_state, reward, terminated , truncated, _ = self.env.step(action)
                if isinstance(state, tuple): 
                    next_state = next_state[0]

                done = terminated or truncated
                self.agent.append_sample(state, action,  reward, next_state, done) 

                state = next_state
                episode_reward += reward
                if done:
                    if self.epsilon_greedy:
                        logger.info("Episode: {}/{}, Episodes reward: {:.6}, e: {:.3}".format(episode, self.cfg.max_episodes, episode_reward, self.epsilon))
                    else:
                        logger.info("Episode: {}/{}, Episodes reward: {:.6f}, t: {:.3f}".format(episode, self.cfg.max_episodes, episode_reward, self.tempertuer))    
                    break

                if total_steps >= self.batch_size :
                    if self.cfg.model_name=="D3QN":
                        loss=self.agent.train_DoubleDQN()
                    else:
                        loss=self.agent.train_DQN()
                    self.agent.update_Gamma()
                    episode_loss +=loss
                    if total_steps % self.target_updates == 0:
                        self.agent._target_update()
                          
            loss_history.append(episode_loss)  
            epsilon_history.append(self.epsilon)
            tempertuer_history.append(self.tempertuer)
            reward_history.append(episode_reward) 
            if self.epsilon_greedy:
                self.epsilon=self.agent.update_epsilon(episode)
            else:
                self.tempertuer=self.agent.update_tempertuer(episode)                                             

            if episode % self.save_interval == 0:
                self.agent.save(self.cfg.save_path + '_' + f'{episode}')
                if self.epsilon_greedy:
                    plot(episode, self.cfg, reward_history,epsilon_history, loss_history)
                else:
                    plot(episode, self.cfg, reward_history,tempertuer_history, loss_history)
                
                                                                
    def test(self, max_episodes):  
        """                
        Reinforcement learning policy evaluation.
        """
           
        # Load the weights of the test_network
        self.agent.dqn.load_state_dict(torch.load(self.RL_load_path))
        self.agent.dqn.eval()
        
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
            
        pygame.quit() 
        
