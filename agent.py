import numpy as np
import der
import env
import torch

class Agent(object):
    def __init__(self):
        pass
        
    def update(self,s,a,r,sp):
        pass
        
    def act(self,s,power_supplied,power_cap,energy_cap):
        pass
    
    def learn(self,k,market,der,episode):
        scores = [] # list containing score from each episode
        scores_window = []
        for i_episode in range(1, self.n_episodes+1):
            state = np.zeros(1) #env.reset()
            score = 0
            energy_cap = der.energy_cap
            power_cap = der.power_cap
            for i in range(k):
                price, power_supplied, done = market.step(i_episode)
                power_supplied = int(power_supplied)
                action = self.act(state,power_supplied,power_cap,energy_cap)
                reward = -1*(action-power_cap)*price
                print(state,action,power_cap, power_supplied)
                next_state = state + (action-power_cap) + power_supplied
                
                self.step(state,action,reward,next_state,done)
                state = next_state
                score += reward
                scores_window.append(score) ## save the most recent score
                scores.append(score) ## sae the most recent score
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
                if i_episode %100==0:
                    print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
                    
                if np.mean(scores_window)>=200.0:
                    print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                            np.mean(scores_window)))
                    torch.save(self.qnetwork_local.state_dict(),'checkpoint.pth')
                    break
        return scores
