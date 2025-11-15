
import numpy as np
import torch
from collections import deque
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
