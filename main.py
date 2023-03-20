import os
import argparse
import torch
import random
from der import DER
from env import Market
from dqn_agent import DQNAgent
from qlearn_agent import q_learning_agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def passed_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--n_epochs', type=int, default=20,
                        help="number of epochs for training")
    parser.add_argument('--gamma', type=float,
                        default=0.95, help="discount factor")
    parser.add_argument('--penalty_factor', type=float,
                        default=4.0, help="discount factor")
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_decay', type=float, default=0.996)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--length_of_day', type=int, default=24)

    args = parser.parse_args()
    return args


def main():
    args = passed_arguments()
    random.seed(args.seed)
    market = Market(1)

    der = DER(power_capacity=args.power_capacity, energy_capacity=args.energy_capacity,
              energy_demand=args.energy_demand, h_demand=args.h_demand)
    gamma = args.gamma

    agent1 = q_learning_agent(nb_states=der.nb_states()*args.length_of_day, nb_actions=der.nb_actions(),
                              gamma=gamma, demand=der.energy_dem, h_demand=der.h_demand, price_penalty=args.penalty_factor,
                              n_epochs=args.n_epochs)
    

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # agent2 = DQNAgent(state_size=1, action_size=der.nb_actions(), gamma=gamma, n_epochs=args.n_epochs,
    #                   eps_start=args.eps_start, eps_decay=args.eps_decay, eps_end=args.eps_end,
    #                   seed=args.seed, demand=der.energy_dem, h_demand=der.h_demand, price_penalty=args.penalty_factor, device=device)

    
    agent = agent1
    a_scores = agent.learn(args.length_of_day, market, der)
    # a1_scores = agent1.learn(args.length_of_day, market, der)
    # a2_scores = agent2.learn(args.length_of_day, market, der)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(list(range(len(a1_scores))), a1_scores, label="Q Learning")
    # ax.plot(list(range(len(a2_scores))), a2_scores, label="DQN")
    # ax.set_ylabel('Score')
    # ax.set_xlabel('Epsiode #')
    # ax.legend()
    # ax.set_title("Training Curves")
    # plt.show()

    # =======================================
    # eval

    # a1_eval_score = agent1.eval(args.length_of_day, market, der)
    # a2_eval_score = agent2.eval(args.length_of_day, market, der)

    # df = pd.DataFrame({'Episode': list(range(args.length_of_day-1))+list(range(args.length_of_day-1)),
    #                   'score': a1_eval_score+a2_eval_score, 'type': ['Q Learning']*(args.length_of_day-1) + ['DQN']*(args.length_of_day-1)})
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # g = sns.barplot(data=df, y='score', x='Episode', hue='type')

    # g.set_ylabel('Eval Score')
    # g.set_xlabel('Epsiode #')
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    # g.set(title='Evaluation Scores')
    # plt.show()

    # Policy of a specific episode
    num_eps_total = len(os.listdir("./data/data_demand"))
    episode_id = random.randint(0, num_eps_total-1)
    print("Episode"+str(episode_id))
    s_list, a_list, ps_list, _, p_list = agent.track_episode(
        24, episode_id, market, der)

    hr_list = list(range(len(s_list)))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('price', color='r')
    ax1.plot(hr_list, p_list, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    #ax2.set_ylabel('sin', color=color)
    bat_lvl_desired = np.minimum(np.ones(24),(np.array(hr_list))/(agent.h_demand-1.0)) * agent.demand    
    surplus = list(np.array(s_list) - bat_lvl_desired)

    ax2.plot(hr_list, ps_list, label='solar power', color='tab:blue')
    ax2.plot(hr_list, a_list, label='actions', color='tab:orange')
    ax2.plot(hr_list, s_list, label='battery level', color='tab:green')
    ax2.plot(hr_list, surplus, label='surplus', color='tab:purple')
    
    fig.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
