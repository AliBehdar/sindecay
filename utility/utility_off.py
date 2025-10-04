
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
        hidden1 = cfg.hidden_size1
        hidden2 = cfg.hidden_size2
        hidden3 = cfg.hidden_size3
        self.num_action = action_size
        self.layer1 = nn.Linear(state_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, hidden3)
        self.value_head = nn.Linear(hidden3,1)
        self.advantage_head = nn.Linear(hidden3,self.num_action)
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
        h = self.activation(self.layer3(h))

        value = self.value_head(h)                  
        advantages = self.advantage_head(h)      
        
        mean_advantages = advantages.mean(dim=1, keepdim=True)  
        centered_advantages = advantages - mean_advantages  

        q_values = value + centered_advantages 

        return q_values
