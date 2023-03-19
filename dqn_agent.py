import random
import torch
import numpy as np
import torch.optim as optim
from replay_buffer import ReplayBuffer
from model import q_network
from agent import Agent

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 32         # minibatch size
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class DQNAgent(Agent):
    """Interacts with and learns form environment."""
    def __init__(self, state_size, action_size, gamma, eps_start, eps_decay, eps_end, seed, 
                 demand, h_demand, price_penalty, n_epochs, device):
        """Initialize an Agent object."""
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.price_penalty = price_penalty
        self.seed = seed
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.demand = demand
        self.gamma = gamma
        self.h_demand = h_demand
        self.n_epochs = n_epochs
        #Q- Network
        self.qnetwork_local = q_network(state_size, action_size, seed).to(device)
        self.qnetwork_target = q_network(state_size, action_size, seed).to(device)     
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed, self.device)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory)>BATCH_SIZE:
                experience = self.memory.sample()
                states, actions, rewards, next_state = experience
                self.update(states, actions, rewards, next_state)
                
    def act(self, state, power_supplied, power_cap, energy_cap):
        """Returns action for given state as per current policy """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > self.eps:
            a = np.argmax(action_values.cpu().data.numpy())
        else:
            a = random.choice(np.arange(self.action_size))

        s = state.item() 
        if s + (a-power_cap) + power_supplied < 0:
            a = power_cap - s - power_supplied
        elif s + (a-power_cap) + power_supplied > energy_cap:
            a = energy_cap + power_cap - s - power_supplied
        
        self.eps = max(self.eps*self.eps_decay, self.eps_end)
        return a
    
    def update(self, states, actions, rewards, next_state):
        """Update value parameters using given batch of experience tuples.
        """
        ## compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train() #train mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval() # eval mode 
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma* labels_next)
        
        loss = criterion(predicted_targets,labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            