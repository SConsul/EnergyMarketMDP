import torch
import numpy as np
from der import DER
from env import Market
from dqn_agent import DQNAgent
from qlearn_agent import q_learning_agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

market = Market(1)

der = DER(power_capacity = 100 ,energy_capacity = 400, energy_demand = 100, h_demand=18)
gamma = 0.95
agent1 = q_learning_agent(nb_states=der.nb_states(),nb_actions=der.nb_actions(),
               gamma=gamma, demand=der.energy_dem,h_demand=der.h_demand, price_penalty=3.0, n_episodes=15)
a1_scores = agent1.learn(24,market,der)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent2 = DQNAgent(state_size=1,action_size=der.nb_actions(),gamma=gamma, n_episodes=15, 
                  eps_start=1.0, eps_decay=0.996, eps_end=0.01, seed= 42,demand=der.energy_dem,
                  h_demand=der.h_demand, price_penalty=3.0, device=device)

a2_scores = agent2.learn(24,market,der)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(a1_scores)),a1_scores, label="Q Learning")
plt.plot(np.arange(len(a2_scores)),a2_scores, label="DQN")
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.legend()
plt.show()

#eval
a1_eval_score = agent1.eval(24,market,der)
a2_eval_score = agent2.eval(24,market,der)
print(len(a1_eval_score+a2_eval_score))
df = pd.DataFrame({'Episode':list(range(23))+list(range(23)),'score':a1_eval_score+a2_eval_score, 'type':['Q Learning']*23 +['DQN']*23})
print(df)
fig = plt.figure()
ax = fig.add_subplot(111)
g = sns.barplot(data=df, y='score', x='Episode', hue='type')

g.set_ylabel('Eval Score')
g.set_xlabel('Epsiode #')
g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.show()