
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
class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consists of Fully Connected (FC) layers with ReLU activation functions.
    """
    
    def __init__(self, num_actions, input_dim):

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

        Q = self.FC(x)    
        return Q
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
            #state_size = (*state_size, 1)
            state_size = (1, *state_size)
        self.in_channels = state_size[0] #
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
        dummy = torch.zeros(1, shape[0], shape[1], shape[2])
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

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Store experiences
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        tree_idx = idx + self.capacity - 1
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(self.write, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def total(self):
        return self.tree[0]

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

    def add(self, experience, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()  # Normalize

        return batch, idxs, is_weights

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)

from collections import deque
class ReplayMemory:
    def __init__(self, capacity,device):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        self.capacity     =    capacity
        self.states       = deque(maxlen=capacity)
        self.actions      = deque(maxlen=capacity)
        self.next_states  = deque(maxlen=capacity)
        self.rewards      = deque(maxlen=capacity)
        self.dones        = deque(maxlen=capacity)
        self.device=device
        
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

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)

        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)

        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)

        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)

        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, dones
    
    
    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque 
        represents the length of the entire memory.
        """
        
        return len(self.dones)
