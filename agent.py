import numpy as np
import der
import env
import torch
import statistics
import random
import seaborn 

class Agent(object):
    def learn(self,k,market,der):

        scores = [] # list containing score from each episode
        episode_list = list(range(23))
        for epoch in range(self.n_epochs):
            random.shuffle(episode_list)
            score = 0.0
            for i_episode in episode_list:
                state = np.zeros(1) #env.reset()
                energy_cap = der.energy_cap
                power_cap = der.power_cap
                market.init_day(i_episode)
                for _ in range(k):
                    price, power_supplied = market.step(i_episode)
                    power_supplied = int(power_supplied)
                    action = self.act(state,power_supplied,power_cap,energy_cap)
                    reward = -1*(action-power_cap)*price

                    if market.t <= self.h_demand and state<self.demand:
                        reward -= (self.demand-state)*price*self.price_penalty*market.t/self.h_demand
                    next_state = state + (action-power_cap) + power_supplied

                    self.step(state,action,reward,next_state)
                    state = next_state
                    score += reward
            scores.append(score) ## save the most recent score

            print('Epoch {}\tWindowed Average Score {:.2f}'.format(epoch,statistics.fmean(scores[-100:])))
        return scores

    def eval(self,k,market,der):
        day_scores = [] # list containing score from each episode
        for i_episode in range(23):
            state = np.zeros(1) #env.reset()
            score = 0.0
            energy_cap = der.energy_cap
            power_cap = der.power_cap
            market.init_day(i_episode)
            for i in range(k):
                price, power_supplied = market.step(i_episode)
                power_supplied = int(power_supplied)
                action = self.act(state,power_supplied,power_cap,energy_cap)
                reward = -1*(action-power_cap)*price
                if market.t <= self.h_demand and state<self.demand:
                    reward -= (self.demand-state)*price*self.price_penalty*market.t/self.h_demand
                next_state = state + (action-power_cap) + power_supplied
                state = next_state
                score += reward
            day_scores.append(score) ## save the most recent score

        print('Average Score {:.2f}'.format(statistics.fsum(day_scores)))

        return day_scores
    
    def track_episode(self, k, episode_id, market, der):
        state = np.zeros(1) #env.reset()
        score = 0.0
        energy_cap = der.energy_cap
        power_cap = der.power_cap
        market.init_day(episode_id)
        
        state_list = []
        action_list = []
        ps_list = []
        reward_list = []
        price_list = []
        for i in range(k):
            price, power_supplied = market.step(episode_id)
            power_supplied = int(power_supplied)
            action = self.act(state,power_supplied,power_cap,energy_cap)
            reward = -1*(action-power_cap)*price
            if market.t <= self.h_demand and state<self.demand:
                reward -= (self.demand-state)*price*self.price_penalty*market.t/self.h_demand
            
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            price_list.append(price)
            ps_list.append(power_supplied)
            next_state = state + (action-power_cap) + power_supplied
            state = next_state
            score += reward

        
        return state_list, action_list, ps_list, reward_list, price_list
