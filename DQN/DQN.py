import os, random 
import gc
import torch
import pygame
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorflow.python.framework import random_seed
device = torch.device("cuda")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Used for debugging; CUDA related errors shown immediately.
seed = 1
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random_seed.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        self.capacity     =    capacity
        self.states       = deque(maxlen=capacity)
        self.actions      = deque(maxlen=capacity)
        self.next_states  = deque(maxlen=capacity)
        self.rewards      = deque(maxlen=capacity)
        self.dones        = deque(maxlen=capacity)
        
        
    def store(self, state, action, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """
        
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        
        
    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """
        
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(device)

        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)

        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)

        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)

        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones
    
    
    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque 
        represents the length of the entire memory.
        """
        
        return len(self.dones)

class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consists of Fully Connected (FC) layers with ReLU activation functions.
    """
    
    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.
        
        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """
        
        super(DQN_Network, self).__init__()
                                                          
        self.FC = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
            )
        
        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
        
    def forward(self, x):
        """
        Forward pass of the network to find the Q-values of the actions.
        
        Parameters:
            x (torch.Tensor): Input tensor representing the state.
        
        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """
        
        Q = self.FC(x)    
        return Q

class DQN_Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """
    
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, 
                  clip_grad_norm, learning_rate, discount,temperature,temperature_decay,min_temperature , memory_capacity):
        
        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
        self.temperature_history=[]
        self.epsilon_history=[]
        # RL hyperparameters
        self.epsilon_max   = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount      = discount

        self.temperature  =temperature 
        self.temperature_decay=temperature_decay 
        self.min_temperature =min_temperature   
        self.action_space  = env.action_space
        self.action_space.seed(seed) # Set the seed to get reproducible results when sampling the action space 
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)
        
        # Initiate the network models
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.shape[0]).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.shape[0]).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
                

    def select_action_g(self, state):
        """
        Selects an action using epsilon-greedy strategy OR based on the Q-values.
        
        Parameters:
            state (torch.Tensor): Input tensor representing the state.
        
        Returns:
            action (int): The selected action.
        """
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()
        
        # Exploitation: the action is selected based on the Q-values.    
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
                        
            return action


    def select_action_b(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        Q_values = self.main_network(state)
        #print("Q-values", Q_values)
        action_probs = F.softmax(Q_values / self.temperature, dim=-1)
        #print("action_probs", action_probs)
        # Sample action from the Boltzmann distribution
        action = torch.multinomial(action_probs, num_samples=1).item()
        #print("action", action)
        return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.
        
        Parameters:
            batch_size (int): The number of experiences to sample from the replay memory.
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """ 
        
        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        actions       = actions.unsqueeze(1)
        rewards       = rewards.unsqueeze(1)
        dones         = dones.unsqueeze(1)       
        
        
        predicted_q = self.main_network(states) # forward pass through the main network to find the Q-values of the states
        predicted_q = predicted_q.gather(dim=1, index=actions) # selecting the Q-values of the actions that were actually taken


        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():            
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0] # not argmax (cause we want the maxmimum q-value, not the action that maximize it)
        next_target_q_value[dones] = 0 # Set the Q-value for terminal states to zero
        y_js = rewards + (self.discount * next_target_q_value) # Compute the target Q-values
        
        
        loss = self.critertion(predicted_q, y_js) # Compute the loss
        
        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts # The average loss for the episode
            self.loss_history.append(episode_loss) # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0
            
        self.optimizer.zero_grad() # Zero the gradients
        loss.backward() # Perform backward pass and update the gradients
        
        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        
        self.optimizer.step() # Update the parameters of the main network using the optimizer
 

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying 
        the parameters from the main network.
        """
        
        self.target_network.load_state_dict(self.main_network.state_dict())
  
    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.
        
        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """
        
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon_max)
    def update_temperature(self):
        """
        Update the value of temperature for Boltzmann exploration.
        
        This method decreases the temperature over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay) 
        self.temperature_history.append(self.temperature)
    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        """
        torch.save(self.main_network.state_dict(), path)
    def update_Gamma(self):
        self.discount_factor = 1 - 0.98 * (1 - self.discount_factor)
        print("Gamma",self.discount_factor)
    
class Model_TrainTest:
    def __init__(self, hyperparams):
        
        # Define RL Hyperparameters
        #self.train_mode             = hyperparams["train_mode"]
        self.epsilon_greedy         = hyperparams["epsilon_greedy"]
        self.RL_load_path           = hyperparams["RL_load_path"]
        self.save_path              = hyperparams["save_path"]
        self.save_interval          = hyperparams["save_interval"]

        self.clip_grad_norm         = hyperparams["clip_grad_norm"]
        self.learning_rate          = hyperparams["learning_rate"]
        self.discount_factor        = hyperparams["discount_factor"]
        self.batch_size             = hyperparams["batch_size"]
        self.update_frequency       = hyperparams["update_frequency"]
        self.max_episodes           = hyperparams["max_episodes"]
        self.max_steps              = hyperparams["max_steps"]
        self.render                 = hyperparams["render"]        
        self.epsilon_max            = hyperparams["epsilon_max"]
        self.epsilon_min            = hyperparams["epsilon_min"]
        self.epsilon_decay          = hyperparams["epsilon_decay"]
        self.temperature            = hyperparams["temperature "]
        self.temperature_decay      = hyperparams["temperature_decay"]
        self.min_temperature        = hyperparams["min_temperature"]
        self.memory_capacity        = hyperparams["memory_capacity"]
        self.render_fps             = hyperparams["render_fps"]
        self.incrising_Ep_Or_temp   = hyperparams["indricing_Ep_Or_temp"]            
        # Define Env
        self.learn_frequency=8
        self.env = gym.make("LunarLander-v2", max_episode_steps=self.max_steps, 
                            render_mode="human" if self.render else None)
        
        self.env.metadata['render_fps'] = self.render_fps # For max frame rate make it 0
        
        # Define the agent class
        self.agent = DQN_Agent( env               = self.env, 
                                epsilon_max       = self.epsilon_max, 
                                epsilon_min       = self.epsilon_min, 
                                epsilon_decay     = self.epsilon_decay,
                                clip_grad_norm    = self.clip_grad_norm,
                                learning_rate     = self.learning_rate,
                                discount          = self.discount_factor,
                                temperature       = self.temperature,
                                temperature_decay = self.temperature_decay,
                                min_temperature   = self.min_temperature,
                                memory_capacity   = self.memory_capacity)
        
    
    def train(self): 
        """                
        Reinforcement learning training loop.
        """
        total_steps = 0
        self.reward_history = []
        
        
        # Training loop over episodes
        for episode in range(1, self.max_episodes+1):
            state, _ = self.env.reset(seed=seed)
            done = False
            
            step_size = 0
            episode_reward = 0
            #and step_size<self.max_steps                                
            while not done :
                if self.epsilon_greedy==True:
                    action = self.agent.select_action_g(state)
                else:
                    action = self.agent.select_action_b(state)
                    
                next_state, reward, done, truncation, _ = self.env.step(action)
                self.agent.replay_memory.store(state, action, next_state, reward, done) 

                if len(self.agent.replay_memory) > self.batch_size and total_steps % self.learn_frequency == 0:
                    self.agent.learn(self.batch_size, (done or truncation))
                
                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()
                
                state = next_state
                episode_reward += reward
                step_size +=1
                            
            # Appends for tracking history
            self.reward_history.append(episode_reward)# episode reward                        
            total_steps += step_size
            #self.agent.update_Gamma()                                                              
            # Decay epsilon at the end of each episode
            if self.incrising_Ep_Or_temp:
                if self.epsilon_greedy:
                    self.agent.update_epsilon()
                else:
                    self.agent.update_temperature()              
            #-- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')
            if self.epsilon_greedy:
                result = (f"Episode: {episode}, "
                        f"Total Steps: {total_steps}, "
                        f"Ep Step: {step_size}, "
                        f"Raw Reward: {episode_reward:.2f}, "
                        f"epsilon: {self.agent.epsilon_max:.2f}")
                print(result)
            else:
                result = (f"Episode: {episode}, "
                        f"Total Steps: {total_steps}, "
                        f"Ep Step: {step_size}, "
                        f"Raw Reward: {episode_reward:.2f}, "
                        f"temperature: {self.agent.temperature:.2f}")
                print(result)
        self.plot_training(episode)
                                                                    

    def test(self, max_episodes):  
        """                
        Reinforcement learning policy evaluation.
        """
           
        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()
        
        # Testing loop over episodes
        for episode in range(1, max_episodes+1):         
            state, _ = self.env.reset(seed=seed)
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
                
    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma_reward = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        # Normalize loss to be in the range from 0 to 300
        normalized_loss = np.interp(self.agent.loss_history, (np.min(self.agent.loss_history), np.max(self.agent.loss_history)), (0, 300))
                

        min_reward=np.min(self.reward_history)
        plt.figure(figsize=(10, 6))

        plt.plot(normalized_loss, label='Normalized Loss', color='#CB291A', alpha=0.8)

        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        
        plt.figure(figsize=(10, 6))

        # Plot Rewards
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=0.8)
        plt.plot(sma_reward, label='SMA 50 Reward', color='#385DAA')
        if self.incrising_Ep_Or_temp:
            if self.epsilon_greedy==True:
                normalized_epsilon = np.interp(self.agent.epsilon_history, (np.min(self.agent.epsilon_history), np.max(self.agent.epsilon_history)), (min_reward/4, 300))
                plt.plot(normalized_epsilon, label='Normalized Epsilon', color='green', alpha=0.8)

        
            else:
                normalized_temperature = np.interp(self.agent.temperature_history, (np.min(self.agent.temperature_history), np.max(self.agent.temperature_history)), (min_reward/4, 300))
                plt.plot(normalized_temperature, label='Normalized Temperature', color='black', alpha=0.8)
 

        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./training_progress.png', format='png', dpi=600, bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()   

if __name__ == '__main__':
    # Parameters:
    train_mode =True
    epsilon_greedy=False
    render = not train_mode
    RL_hyperparams = {
        "train_mode"            : train_mode,
        "epsilon_greedy"        : epsilon_greedy,
        "RL_load_path"          : f'./weights-and-plot/final_weights' + '_' + '600' + '.pth',
        "save_path"             : f'./weights-and-plot/final_weights',
        "save_interval"         : 100,
        "render_fps"            : 6,
        "render"                : render,
        "indricing_Ep_Or_temp"  : True,
        "temperature "          : 1,
        "min_temperature"       : 0.01,
        "temperature_decay"     : 0.99,
        
        "memory_capacity"       : 100000        if train_mode else 0,
        "max_steps"             : 1000          if train_mode else 500,
        "batch_size"            : 128,
        "update_frequency"      : 40,
        "max_episodes"          : 500          if train_mode else 5,
        "clip_grad_norm"        : 3,

        "learning_rate"         : 0.001,
        "discount_factor"       : 0.993,
        "epsilon_max"           : 0.99         if train_mode else -1,
        "epsilon_min"           : 0.01,
        "epsilon_decay"         : 0.99,
        

        }
    
    
    # Run
    DRL = Model_TrainTest(RL_hyperparams) # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])