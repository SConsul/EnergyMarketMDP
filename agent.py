import numpy as np
import der
import env

class Agent(object):
    def __init__(self,nb_states,nb_actions,demand,h_demand):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.demand = demand
        self.h_demand = h_demand
        self.Q = np.zeros((nb_states,nb_actions))
        self.policy = np.zeros(nb_states)
        self.gamma = 0.95
        self.alpha = 0.2
        self.eps = 0.1
        
    def update_Q_learning(self,s,a,r,sp):
        self.Q[s,a] += self.alpha*(r + self.gamma*max(self.Q[sp,:]) - self.Q[s, a])
        
    def greedy_policy(self,s,power_supplied,power_cap,energy_cap):
        rand_v = np.random.uniform(0,1)
        if rand_v < self.eps:
            self.policy[s] = np.random.randint(0,self.nb_actions)
        else:
            self.policy[s] = np.argmax(self.Q[s])
        if s + (self.policy[s]-power_cap) + power_supplied < 0:
            self.policy[s] = power_cap - s - power_supplied
        if s + (self.policy[s]-power_cap) + power_supplied > energy_cap:
            self.policy[s] = energy_cap + power_cap - s - power_supplied
            print('too big')
    
    def Q_learning(self,k,market,der,episode):
        s = 0
        list_hour = []
        list_price = []
        list_power_supplied = []
        list_actions = []
        list_s = []
        energy_cap = der.energy_cap
        power_cap = der.power_cap
        for i in range(k):
            price, power_supplied, t0 = market.step(episode)
            power_supplied = int(power_supplied)
            self.greedy_policy(s,power_supplied,power_cap,energy_cap)
            a = self.policy[s]
            r = -1*(a-der.power_cap)*price
            if t0 <= self.h_demand:
                if s<self.demand:
                    r-=(self.demand-s)*price*2*t0/self.h_demand
            sp = s + (a-der.power_cap) + power_supplied
            a = int(a)
            sp = int(sp)
            self.update_Q_learning(s,a,r,sp)
            list_hour.append(market.t)
            list_price.append(price)
            list_power_supplied.append(power_supplied)
            list_actions.append(a-der.power_cap)
            list_s.append(s)
            s = sp
        return list_hour, list_price, list_power_supplied, list_actions, list_s
