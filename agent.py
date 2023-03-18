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
        scores_window = []
        episode_list = list(range(23))
        random.shuffle(episode_list)
        if self.n_episodes<23:
            episode_list = episode_list[:self.n_episodes]
        else:
            episode_list = episode_list + random.sample(list(range(23)),self.n_episodes-23)
        for idx, i_episode in enumerate(episode_list):
            state = np.zeros(1) #env.reset()
            score = 0
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
                
                self.step(state,action,reward,next_state,0)
                state = next_state
                score += reward
                scores_window.append(score) ## save the most recent score
                scores.append(score) ## save the most recent score

                print('Episode {}\tAverage Score {:.2f}'.format(idx,statistics.fmean(scores_window)))
                if i_episode %100==0:
                    print('Episode {}\tAverage Score {:.2f}'.format(idx,statistics.fmean(scores_window)))
                    
                if statistics.fmean(scores_window)>=200.0:
                    print('Environment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(idx-100,
                                                                                            statistics.fmean(scores_window)))
                    torch.save(self.qnetwork_local.state_dict(),'checkpoint.pth')
                    break
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