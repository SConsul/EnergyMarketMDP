import torch
import numpy as np
import random 
from collections import namedtuple, deque 


class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object. """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state"])
        self.seed = torch.manual_seed(seed)
        self.device = device

    def add(self,state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        states = torch.from_numpy(np.vstack([np.array([e.state[0],e.state[1]]) for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states,actions,rewards,next_states)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)