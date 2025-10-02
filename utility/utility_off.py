import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import os,random
from tensorflow.python.framework import random_seed
def seed(cfg):
    np.random.seed(cfg.seed)
    np.random.default_rng(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random_seed.set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

class Network(Model):
    def __init__(self, state_size: int, action_size: int,cfg ):
        """
        Initialization.
        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        :param hidden_size: The size of the hidden layers.
        """
        super(Network, self).__init__()
        
        self.num_action = action_size
        self.layer1 = tf.keras.layers.Dense(cfg.hidden_size, activation='relu')# Define the first hidden layer with ReLU activation
        self.layer2 = tf.keras.layers.Dense(cfg.hidden_size, activation='relu')# Define the second hidden layer with ReLU activation
        self.state = tf.keras.layers.Dense(self.num_action)# Define the output layer for state values
        self.action = tf.keras.layers.Dense(self.num_action)# Define the output layer for action values

    def call(self, state):
        """
        Forward pass of the network.
        :param state: Input state.
        :return: Value function Q(s, a).
        """
        layer1 = self.layer1(state) # Pass the input state through the first hidden layer      
        layer2 = self.layer2(layer1)  # Pass the result through the second hidden layer
        state = self.state(layer2) # Compute the state values       
        action = self.action(layer2) # Compute the action values        
        mean = tf.keras.backend.mean(action, keepdims=True)# Calculate the mean of the action values 
        advantage = (action - mean)# Calculate the advantage by subtracting the mean action value      
        value = state + advantage # Compute the final Q-values by adding state values and advantages 

        return value