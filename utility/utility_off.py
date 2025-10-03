
import numpy as np
import os,random
import torch
import torch.nn as nn
import torch.nn.functional as F

def seed(cfg):
    s =int(cfg.seed)
    np.random.seed(s)
    np.random.default_rng(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(cfg.seed)
    
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Deterministic behavior for cuDNN (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # can raise errors on non-deterministic ops
    # torch.use_deterministic_algorithms(True)
class Network(nn.Module):
    def __init__(self, state_size: int, action_size: int,cfg=None ):
        """
        PyTorch version of your network.
        :param state_size: input dimension (not used explicitly in linear layers here,
                           but kept for API compatibility)
        :param action_size: number of actions (output dimension)
        :param cfg: object with attribute hidden_size
        """

        super().__init__()
        hidden = cfg.hidden_size if cfg is not None else 128
        self.num_action = action_size
        self.layer1 = nn.Linear(state_size, hidden)# Define the first hidden layer with ReLU activation
        self.layer2 = nn.Linear(hidden, hidden)# Define the second hidden layer with ReLU activation
        self.state_head = nn.Linear(hidden,self.num_action)# Define the output layer for state values
        self.action_head = nn.Linear(hidden,self.num_action)# Define the output layer for action values
        self.activation = nn.ReLU()

    def forward(self, state):
        """
        Forward pass.
        x: tensor of shape (batch, state_size) or (state_size,) for single sample
        returns: Q-values tensor of shape (batch, num_action)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        h = self.activation(self.layer1(state))
        h = self.activation(self.layer2(h))
        
        state_vals = self.state_head(h)   # (batch, num_action)
        action_vals = self.action_head(h)       
        
        mean = state_vals = self.state_head(h)   # (batch, num_action)

        advantage = action_vals - mean    # Calculate the advantage by subtracting the mean action value      
        q_values = state_vals + advantage # Compute the final Q-values by adding state values and advantages 

        return q_values