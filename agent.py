import numpy as np
import der
import env
import torch
import statistics
import random
import seaborn 
import math 
class Agent(object):

    #State Index bat_lvl [0,399], hr [0 23] -> hr*400 + bat_lvl [0 9599]

    def learn(self,k,market,der):

        scores = [] # list containing score from each episode
        episode_list = list(range(23))
        for epoch in range(self.n_epochs):
            random.shuffle(episode_list)
            score = 0.0
            for i_episode in episode_list:
                bat_lvl = np.zeros(1) #env.reset()
                hr = np.zeros(1)
                energy_cap = der.energy_cap
                power_cap = der.power_cap
                market.init_day(i_episode)
                for _ in range(k):
                    hr = market.t
                    state = hr*400+bat_lvl
                    price, power_supplied = market.step(i_episode)
                    power_supplied = int(power_supplied)
                    action = self.act(hr, bat_lvl ,power_supplied, power_cap,energy_cap, self.demand)
                    reward = -1*(action-power_cap)*price

                    if market.t <= self.h_demand and bat_lvl<self.demand:
                        reward -= (self.demand-bat_lvl)*price*self.price_penalty*market.t/self.h_demand
                    new_bat_lvl = bat_lvl + (action-power_cap) + power_supplied
                    next_state = ((hr+1)%24)*400+new_bat_lvl
                    # if bat_lvl<self.demand:
                    #     reward -= (self.demand-bat_lvl)*price*self.price_penalty
                    #     min_action = int(math.ceil((self.demand-bat_lvl)+power_cap))
                    #     for a in range(min_action+1):
                    #         new_bat_lvl = bat_lvl + (action-power_cap) + power_supplied - self.demand
                    #         next_state = ((hr+1)%24)*400+new_bat_lvl
                    #         self.step(state,a,reward,next_state)
                    # else:
                    #     new_bat_lvl = bat_lvl + (action-power_cap) + power_supplied - self.demand
                    #     next_state = ((hr+1)%24)*400+new_bat_lvl
                    #     print("next state=", next_state, "  (hr,bt) =", (hr+1)%24, new_bat_lvl,)
                    #     self.step(state,action,reward,next_state)
                    self.step(state,action,reward,next_state)
                    bat_lvl = new_bat_lvl
                    score += reward
            scores.append(score) ## save the most recent score
            
            print('Epoch {}\tWindowed Average Score {:.2f}'.format(epoch,statistics.fmean(scores[-100:])))
        return scores

    def eval(self,k,market,der):
        day_scores = [] # list containing score from each episode
        for i_episode in range(23):
            bat_lvl = np.zeros(1) #env.reset()
            score = 0.0
            energy_cap = der.energy_cap
            power_cap = der.power_cap
            market.init_day(i_episode)
            for _ in range(k):
                hr = market.t
                state = hr*400+bat_lvl
                price, power_supplied = market.step(i_episode)
                power_supplied = int(power_supplied)
                action = self.act(hr, bat_lvl ,power_supplied, power_cap,energy_cap, self.demand)
                reward = -1*(action-power_cap)*price

                if market.t <= self.h_demand and bat_lvl<self.demand:
                    reward -= (self.demand-bat_lvl)*price*self.price_penalty*market.t/self.h_demand
                new_bat_lvl = bat_lvl + (action-power_cap) + power_supplied
                bat_lvl = new_bat_lvl
                score += reward
            day_scores.append(score) ## save the most recent score

        print('Average Score {:.2f}'.format(statistics.fsum(day_scores)))

        return day_scores
    
    def track_episode(self, k, episode_id, market, der):
        bat_lvl = np.zeros(1) #env.reset()
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
            hr = market.t
            price, power_supplied = market.step(episode_id)
            power_supplied = int(power_supplied)
            action = self.act(hr, bat_lvl ,power_supplied, power_cap,energy_cap, self.demand)
            reward = -1*(action-power_cap)*price
            if market.t <= self.h_demand and bat_lvl<self.demand:
                reward -= (self.demand-bat_lvl)*price*self.price_penalty*market.t/self.h_demand

            state_list.append(bat_lvl[0])
            action_list.append(action-power_cap)
            reward_list.append(reward)
            price_list.append(price)
            ps_list.append(power_supplied)

            new_bat_lvl = bat_lvl + (action-power_cap) + power_supplied
            bat_lvl = new_bat_lvl
            score += reward

        
        return state_list, action_list, ps_list, reward_list, price_list


# a_idx -> [0 200] #nA = 201
# a_val -> -100 100 #201 values
# a_val= a_idx - 100
