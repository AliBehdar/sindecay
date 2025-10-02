import pygame,random,gymnasium as gym,numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import hydra 
from omegaconf import DictConfig
from collections import deque
import sys
import os
# Add the parent directory (containing 'utility') to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utility.plots import plot
from utility.utility_off import seed,Network
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
        self.cfg=cfg
        # CREATING THE Q-Network
        self.env = env
        self.lr=cfg.learning_rate
        self.action_space = env.action_space
        self.action_space.seed(self.cfg.seed)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.dqn = Network(self.state_size, self.action_size,cfg)
        self.dqn_target = Network(self.state_size, self.action_size,cfg)
        self.optimizers = optimizers.Adam(learning_rate=self.lr)
        self.memory = deque(maxlen=self.cfg.memory_size)
        self.Soft_Update = False # use soft parameter update
        self._target_hard_update()
        self.loss_history=[]
    # EXPLORATION VS EXPLOITATION
    def get_action(self, state, epsilon):
        q_value = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        # Choose an action a in the current world state (s)
        # If this number < greater than epsilon doing a random choice --> exploration
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)

        ## Else --> exploitation (taking the biggest Q value for this state)
        else:
            action = np.argmax(q_value) 
        return action
    
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    # UPDATING THE Q-VALUE
    def train_step(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states      = [i[0] for i in mini_batch]
        actions     = [i[1] for i in mini_batch]
        rewards     = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones       = [i[4] for i in mini_batch]
        
        dqn_variable = self.dqn.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            states      = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
            actions     = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)
            dones       = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            next_Qs = self.dqn(next_states)
            next_Qs = tf.stop_gradient(next_Qs)
            next_Q_targs = self.dqn_target(next_states)
            next_action = tf.argmax(next_Qs, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * next_Q_targs, axis=1)
            
            mask = 1 - dones
            target_value = rewards + self.gamma * target_value * mask 
            
            curr_Qs = self.dqn(states)
            
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * curr_Qs, axis=1)
            error = tf.square(main_value - target_value) * 0.5
            loss  = tf.reduce_mean(error)
            self.loss_history.append(loss)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))
        
    # after some time interval update the target model to be same with model
    def _target_hard_update(self):
        if not self.Soft_Update:
            self.dqn_target.set_weights(self.dqn.get_weights())
            return
        if self.Soft_Update:
            q_model_theta = self.dqn.get_weights()
            dqn_target_theta = self.dqn_target.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, dqn_target_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                dqn_target_theta[counter] = target_weight
                counter += 1
            self.dqn_target.set_weights(dqn_target_theta)
    
    def update_Gamma(self):
        self.gamma = 1 - 0.985 * (1 - self.gamma)
    def load(self, phat):
        
        self.dqn = tf.keras.models.load_model(phat, custom_objects={'Network': Network})
    def save(self, phat):
        self.dqn.save(phat)


@hydra.main(version_base="1.1", config_path="./conf", config_name="configs")
def main(cfg: DictConfig):
    seed(cfg)
    env = gym.make("LunarLander-v2",render_mode="human" if not cfg.train else None)
    agent = DQNAgent(env,cfg)
    if cfg.train:
        update_cnt    = 0
        scores = []

        reward_history=[]
        epsilon_history=[]   
        for episode in range(1,cfg.max_episodes+1):
            state = agent.env.reset(seed=1)
            state=state[0]
            episode_reward = 0
            done = False  
            while not done :
                update_cnt += 1
                action = agent.get_action(state, cfg.epsilon)
                next_state, reward, done, _ ,_= agent.env.step(action)
                if isinstance(state, tuple): 
                        next_state = next_state[0]
                agent.append_sample(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # if episode ends
                if done:
                    scores.append(episode_reward)
                    print("episode: {}/{}, score: {}, e: {:.4}".format(episode+1, cfg.max_episodes, episode_reward, cfg.epsilon)) 
                    break

                if (update_cnt >= agent.batch_size):
                    agent.train_step()
                    if update_cnt % agent.target_update == 0:
                        agent._target_hard_update()
            
            reward_history.append(episode_reward)   
            epsilon_history.append(cfg.epsilon)
            cfg.epsilon = cfg.min_epsilon + (cfg.max_epsilon - cfg.min_epsilon)*np.exp(-cfg.decay_rate*episode) 
            
            if episode % cfg.save_intervalve==0:
                agent.save(cfg.save_path + '_' + f'{episode}')
                plot(episode,cfg,reward_history,epsilon_history,agent.loss_history)

    else:
        agent.load(cfg.load_path)
        scores = []
        for episode in range(5):
            state = agent.env.reset(seed=1)
            state=state[0]
            episode_reward = 0
            done = False  
            while not done:
                action = agent.get_action(state,0.01)
                next_state, reward, done, _ ,_= agent.env.step(action)
                if isinstance(state, tuple): 
                        next_state = next_state[0]
                agent.append_sample(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    scores.append(episode_reward)
                    print("episode: {}/{}, score: {}, e: {:.4}".format(episode+1, cfg.max_episodes, episode_reward, 0.01)) 
                    break
        pygame.quit()

if __name__=="__main__":
    main()