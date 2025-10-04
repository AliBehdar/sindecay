
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # can raise errors on non-deterministic ops
    # torch.use_deterministic_algorithms(True)
class NetworkMLP(nn.Module):
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
class NetworkCNN(nn.Module):
    def __init__(self, state_size: int, action_size: int, cfg=None):
        
        """
        PyTorch version of your network.
        :param state_size: input dimension (not used explicitly in linear layers here,but kept for API compatibility)
        :param action_size: number of actions (output dimension)
        :param cfg: object with attribute hidden_size
        """

        super().__init__()
        self.activation = nn.ReLU()
        if len(state_size) == 2:
            state_size = (*state_size, 1)
        self.in_channels = state_size[2]
        hidden1 = cfg.hidden_size1
        hidden2 = cfg.hidden_size2
        hidden3 = cfg.hidden_size3
        self.conv1 = nn.Conv2d(self.in_channels,hidden1, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(hidden2, hidden3, kernel_size=3, stride=1)
        
        self.flatten_size = self._get_flatten_size(state_size)
        self.fc = nn.Linear(self.flatten_size, hidden3)

        self.value_head = nn.Linear(hidden3, 1)
        self.advantage_head = nn.Linear(hidden3, action_size)
        
        
    def _get_flatten_size(self, shape):
        # shape = (H, W, C); dummy = (1, C, H, W)
        dummy = torch.zeros(1, shape[2], shape[0], shape[1])
        h = self.activation(self.conv1(dummy))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        return int(np.prod(h.shape[1:]))
    def forward(self, state):
        """
        Forward pass.
        x: tensor of shape (batch, state_size) or (state_size,) for single sample
        returns: Q-values tensor of shape (batch, num_action)
        """
        if state.dim() == 3:  # Single sample without channel (e.g., grayscale)
            state = state.unsqueeze(0)
        if len(state.shape) == 3:  # (batch, h, w) for grayscale, add channel
            state = state.unsqueeze(1)  # (batch, 1, h, w)
        elif len(state.shape) == 4 and state.shape[-1] in [1, 3]:  # (batch, h, w, c)
            state = state.permute(0, 3, 1, 2)  # (batch, c, h, w) 
        
        h = self.activation(self.conv1(state))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        h = h.flatten(start_dim=1)
        h = self.activation(self.fc(h))

        value = self.value_head(h)
        advantages = self.advantage_head(h)
        
        mean_advantages = advantages.mean(dim=1, keepdim=True)
        centered_advantages = advantages - mean_advantages
        
        q_values = value + centered_advantages

        return q_values