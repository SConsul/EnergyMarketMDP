import numpy as np
from agent import Agent
import random

class base_agent(Agent):
    def __init__(self,nb_states,nb_actions,gamma, demand, h_demand, price_penalty, n_epochs):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.demand = demand
        self.h_demand = h_demand
        self.price_penalty = price_penalty
        self.Q = np.zeros((nb_states,nb_actions))
        self.gamma = gamma
        self.alpha = 0.2
        self.eps = 0.1
        self.n_epochs = n_epochs
    
    def step(self, state, action, reward, next_state):
        pass

    def act(self, hr, bat_lvl, power_supplied,power_cap,energy_cap, demand):
        a = min(power_cap - bat_lvl - power_supplied + demand, 
                energy_cap + power_cap - bat_lvl - power_supplied + demand)

        return a
