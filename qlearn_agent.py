import numpy as np
from agent import Agent
import random

class q_learning_agent(Agent):
    def __init__(self,nb_states,nb_actions,gamma, demand, h_demand, price_penalty, n_epochs):
        self.nb_states = nb_states-1
        self.nb_actions = nb_actions-1
        self.demand = demand
        self.h_demand = h_demand
        self.price_penalty = price_penalty
        self.Q = np.zeros((nb_states,nb_actions))
        self.gamma = gamma
        self.alpha = 0.2
        self.eps = 0.1
        self.n_epochs = n_epochs

    def update(self,s,a,r,sp):
        self.Q[s,a] += self.alpha*(r + self.gamma*max(self.Q[sp,:]) - self.Q[s, a])
    
    def step(self, state, action, reward, next_state):
        self.update(int(state[0]), action, reward, int(next_state[0]))

    def act(self,s,power_supplied,power_cap,energy_cap):
        rand_v = np.random.uniform(0,1)
        s = int(s[0])
        if rand_v < self.eps:
            a = random.randint(0,self.nb_actions)
        else:
            a = np.argmax(self.Q[s])
        if s + (a-power_cap) + power_supplied < 0:
            a = power_cap - s - power_supplied
        if s + (a-power_cap) + power_supplied > energy_cap:
            a = energy_cap + power_cap - s - power_supplied
        return a
