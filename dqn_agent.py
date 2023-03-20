import random
import torch
import numpy as np
import torch.optim as optim
from replay_buffer import ReplayBuffer
from model import q_network
from agent import Agent

class DQNAgent(Agent):
    """Interacts with and learns form environment."""
    def __init__(self, state_dim, action_size, gamma, eps_start, eps_decay, eps_end, seed, 
                 demand, h_demand, price_penalty, n_epochs, device, buffer_size=int(1e5), 
                 batch_size=32, tau=1e-3, lr=5e-4, update_freq=4):
        """Initialize an Agent object."""
        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.seed = seed
        self.demand = demand
        self.h_demand = h_demand
        self.price_penalty = price_penalty
        self.n_epochs = n_epochs
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq

        #Q- Network
        self.qnetwork_local = q_network(state_dim, action_size, seed).to(device)
        self.qnetwork_target = q_network(state_dim, action_size, seed).to(device)     
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size,seed, self.device)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory)>self.batch_size:
                experience = self.memory.sample()
                states, actions, rewards, next_state = experience
                self.update(states, actions, rewards, next_state)

    def act(self, hr, bat_lvl, power_supplied, power_cap, energy_cap, demand):
        """Returns action for given state as per current policy """
        state = torch.Tensor([hr/24.0,bat_lvl/400.0]).unsqueeze(0).to(self.device) #normalize state for stablity
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() < self.eps:
            a = random.randint(0,self.action_size-1) #random.choice(np.arange(self.action_size))
        else:
            a = np.argmax(action_values.cpu().data.numpy())
        
        self.eps = max(self.eps*self.eps_decay, self.eps_end)
        return a
    
    def update(self, states, actions, rewards, next_state):
        """Update value parameters using given batch of experience tuples."""
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train() #train mode
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval() # eval mode 
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max()

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma* labels_next)
        
        loss = criterion(predicted_targets,labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.tau)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            