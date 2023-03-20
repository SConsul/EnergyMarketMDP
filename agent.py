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
    def run(self, k, market, der, epoch_freq, train=True, episode_id=None):

        epoch_scores = [] # list containing score from each episode
        episode_list = list(range(23))
        if train:
            num_epochs = self.n_epochs 
        else:
            num_epochs = 1
            if episode_id:
                state_list = []
                action_list = []
                ps_list = []
                reward_list = []
                price_list = []
                episode_list = [episode_id]
        for epoch in range(num_epochs):
            if train:
                random.shuffle(episode_list)
            scores = [] # list containing score from each episode
            score = 0.0
            for i_episode in episode_list:
                bat_lvl = 0.0
                hr = 0.0
                energy_cap = der.energy_cap
                power_cap = der.power_cap
                market.init_day(i_episode)
                for _ in range(k):
                    hr = market.t
                    state = hr*400+bat_lvl
                    price, power_supplied = market.step(i_episode)
                    power_supplied = int(power_supplied)
                    action = self.act(hr, bat_lvl ,power_supplied, power_cap,energy_cap, self.demand*min(1.0,(hr+1)/self.h_demand))
                    reward = -1*(action-power_cap)*price

                    if hr <= self.h_demand and bat_lvl<self.demand*min(1.0,hr/self.h_demand):
                        reward -= (self.demand-bat_lvl)*price*self.price_penalty*hr/self.h_demand
                    new_bat_lvl = min(energy_cap, max(0,bat_lvl + (action-power_cap) + power_supplied))
                    new_hr = (hr+1)%24

                    if train:
                        self.step((hr, bat_lvl) ,action,reward,(new_hr,new_bat_lvl))
                    elif episode_id:
                        state_list.append(bat_lvl)
                        action_list.append(action-power_cap)
                        reward_list.append(reward)
                        price_list.append(price)
                        ps_list.append(power_supplied)
                    bat_lvl = new_bat_lvl
                    score += reward
                    
                scores.append(score) ## save the most recent score
                if episode_id is not None:
                    return state_list, action_list, ps_list, reward_list, price_list
            epoch_score = statistics.fmean(scores)
            epoch_scores.append(epoch_score)
            
            if (epoch+1)%epoch_freq==0:
                if train:
                    print('Epoch {}\t Average Score {:.2f}'.format(epoch+1,epoch_score))
                else:
                    print('Average Score {:.2f}'.format(epoch_score))
        if train:
            return epoch_scores
        return scores

# a_idx -> [0 200] #nA = 201
# a_val -> -100 100 #201 values
# a_val= a_idx - 100
