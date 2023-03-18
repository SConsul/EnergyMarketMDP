import torch
import numpy as np
from der import DER
from env import Market
from dqn_agent import DQNAgent
from qlearn_agent import q_learning_agent
import matplotlib.pyplot as plt

market1 = Market(1)

der1 = DER(power_capacity = 100 ,energy_capacity = 400, energy_demand = 100, h_demand=18)
gamma = 0.95
# agent1 = q_learning_agent(nb_states=der1.nb_states(),nb_actions=der1.nb_actions(),
#                gamma=gamma, demand=der1.energy_dem,h_demand=der1.h_demand)
# list_hour, list_price, list_power_supplied, list_actions, list_s = agent1.learn(24,market1,der1,1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent2 = DQNAgent(state_size=1,action_size=der1.nb_actions(),gamma=gamma, n_episodes=15, eps_start=1.0, eps_decay=0.996, eps_end=0.01, seed= 42,demand=der1.energy_dem,h_demand=der1.h_demand, device=device)

scores = agent2.learn(24,market1,der1,1)[-100:]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()