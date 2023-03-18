import numpy as np
from agent import Agent

class q_learning_agent(Agent):
    def __init__(self,nb_states,nb_actions,gamma, demand,h_demand):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.demand = demand
        self.h_demand = h_demand
        self.Q = np.zeros((nb_states,nb_actions))
        self.gamma = gamma
        self.alpha = 0.2
        self.eps = 0.1
        
    def update(self,s,a,r,sp):
        self.Q[s,a] += self.alpha*(r + self.gamma*max(self.Q[sp,:]) - self.Q[s, a])
        
    def act(self,s,power_supplied,power_cap,energy_cap):
        rand_v = np.random.uniform(0,1)
        if rand_v < self.eps:
            a = np.random.randint(0,self.nb_actions)
        else:
            a = np.argmax(self.Q[s])
        if s + (a-power_cap) + power_supplied < 0:
            a = power_cap - s - power_supplied
        if s + (a-power_cap) + power_supplied > energy_cap:
            a = energy_cap + power_cap - s - power_supplied
            print('too big')
        return a