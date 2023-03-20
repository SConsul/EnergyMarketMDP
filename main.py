import os
import argparse
import torch
import random
from der import DER
from env import Market
from dqn_agent import DQNAgent
from qlearn_agent import q_learning_agent
from base_agent import base_agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def passed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_id', type=int,
                        default=1, help="(1) Q Learning (2)DQN (3)Baseline")
    parser.add_argument('--power_capacity', type=int,
                        default=100, help="Power Capacity")
    parser.add_argument('--energy_capacity', type=int,
                        default=400, help="Energy Capacity")
    parser.add_argument('--energy_demand', type=int,
                        default=100, help="Energy Demand")
    parser.add_argument('--h_demand', type=int, default=18,
                        help="Hour demand is checked")
    parser.add_argument('--seed', type=int, default=42,
                        help="seed for replicating randomness")
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help="number of epochs for training")
    parser.add_argument('--epoch_freq', type=int, default=100,
                        help="frequency of printing logs in training")
    parser.add_argument('--gamma', type=float,
                        default=0.95, help="discount factor")
    parser.add_argument('--penalty_factor', type=float,
                        default=4.0, help="discount factor")
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_decay', type=float, default=0.996)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--length_of_day', type=int, default=24)
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument('--batch_size', type=int, default=32, help="minibatch size for DQN")
    parser.add_argument('--tau', type=float, default=1e-3, help="for soft update of target parameters")
    parser.add_argument('--lr', type=float, default=5e-4, help="learning rate for DQN")
    parser.add_argument('--update_freq', type=int, default=4, help="how often to update the network")

    args = parser.parse_args()
    return args


def main():
    args = passed_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    market = Market(args.seed)

    der = DER(power_capacity=args.power_capacity, energy_capacity=args.energy_capacity,
              energy_demand=args.energy_demand, h_demand=args.h_demand)
    gamma = args.gamma

    if args.agent_id==1:
        agent = q_learning_agent(nb_states=der.nb_states()*args.length_of_day, nb_actions=der.nb_actions(),
                              gamma=gamma, demand=der.energy_dem, h_demand=der.h_demand, price_penalty=args.penalty_factor,
                              n_epochs=args.n_epochs)
    elif args.agent_id==2:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(state_dim=2, action_size=der.nb_actions(), gamma=gamma, n_epochs=args.n_epochs,
                      eps_start=args.eps_start, eps_decay=args.eps_decay, eps_end=args.eps_end,
                      seed=args.seed, demand=der.energy_dem, h_demand=der.h_demand, price_penalty=args.penalty_factor, 
                      device=device, buffer_size=args.buffer_size, batch_size=args.batch_size, tau=args.tau, lr=args.lr, update_freq=args.update_freq)

    elif args.agent_id==3:
        agent = base_agent(nb_states=der.nb_states()*args.length_of_day, nb_actions=der.nb_actions(),
                              gamma=gamma, demand=der.energy_dem, h_demand=der.h_demand, price_penalty=args.penalty_factor,
                              n_epochs=args.n_epochs)

    a_scores = agent.run(args.length_of_day, market, der, args.epoch_freq, train=True)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list(range(len(a_scores))), a_scores)
    ax.set_ylabel('Score of Agent {}'.format(args.agent_id))
    ax.set_xlabel('Episode #')
    ax.legend()
    ax.set_title('Training curve for Agent {}'.format(args.agent_id))
    plt.show()
    
    num_eps_total = len(os.listdir("./data/data_demand"))
    episode_id = random.randint(0, num_eps_total-1)
    s_list, a_list, ps_list, _, p_list = agent.run(args.length_of_day, market, der, args.epoch_freq, False, episode_id)
    hr_list = list(range(len(s_list)))
    bat_lvl_desired = np.minimum(np.ones(24),(np.array(hr_list))/(agent.h_demand)) * agent.demand    
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('price', color='r')
    ax1.plot(hr_list, p_list, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    #ax2.set_ylabel('sin', color=color)
    
    surplus = list(np.array(s_list) - bat_lvl_desired)

    ax2.plot(hr_list, ps_list, label='Solar Power', color='tab:blue')
    ax2.plot(hr_list, a_list, label='Action', color='tab:orange')
    ax2.plot(hr_list, s_list, label='Battery Level', color='tab:green')
    ax2.plot(hr_list, surplus, label='Surplus', color='tab:purple')
    fig.suptitle('Performance of Agent {} on Episode {}'.format(args.agent_id, episode_id))
    fig.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
