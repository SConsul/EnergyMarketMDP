# EnergyMarketMDP
This repository is part of the final project for Stanford AA228 (Winter 2023), taught by [Mykel Kochenderfer](https://mykel.kochenderfer.com/).

Through this code, you will be able to formulate Energy Markets as an MDP and solve it using RL.

### Clone the repository
Clone this repository using:

```bash
git clone https://github.com/SConsul/EnergyMarketMDP.git
```

## Installation
To use this script, you will need to have Python 3.x and use the below command to install all the necessary packages:

```pip install -r requirements.txt```


## Running the Script 
```
python main.py [--agent_id] [--power_capacity] [--energy_capacity] [--energy_demand] [--h_demand] [--seed] [--n_epochs] [--epoch_freq] [--gamma] [--penalty_factor] [--eps_start] [--eps_decay] [--eps_end] [--length_of_day] [--buffer_size] [--batch_size] [--tau] [--lr] [--update_freq]
```

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|--agent_id| 1 | (int) Choose the type of reinforcement learning agent you want to use (1)Q Learning (2) DQN (3) Baseline|
|--power_capacity| 100 | (int) Set the power capacity for the energy management system.
|--energy_capacity | 400 | (int) Set the energy capacity for the energy management system.|
|--energy_demand | 100 | (int) Set the energy demand for the energy management system.|
|--h_demand | 18| (int) Set the hour of the day when the demand is checked.|
|--seed | 42 | (int) Set the seed for replicating randomness. |
|--n_epochs | 1000 | (int) Set the number of epochs for training. |
|--epoch_freq| 100 | (int) Set the frequency of printing logs in training. |
|--gamma | 0.95 | (float) Set the discount factor.|
|--penalty_factor: | 4.0 | (float) Set the penalty factor.|
|--eps_start: | 1.0 | (float) Set the starting value of epsilon for the epsilon-greedy policy. |
|--eps_decay| 0.996| (float) Set the decay rate of epsilon for the epsilon-greedy policy.|
|--eps_end| 0.01 | (float) Set the minimum value of epsilon for the epsilon-greedy policy.|
|--length_of_day| 24 | (int) Set the length of a day in hours.|
|--buffer_size |100000| (int) Set the size of the replay buffer for the DQN agent.|
|--batch_size| 32| (int) Set the minibatch size for the DQN agent.|
|--tau| 0.001 | (float) Set the value of tau for the soft update of target parameters in the DQN agent. |
|--lr: | 0.005| (float) Set the learning rate for the DQN agent.| 
|--update_freq | 4 | Set the frequency of updating the network in the DQN agent.|


The script will run and output the results of the energy management system.


**Team Members:** Paula Charles ([**@paula-charles**](https://github.com/paula-charles)), Sarthak Consul ([**@SConsul**](https://github.com/SConsul)), Nidhi Baid ([**@nidhibaid**](https://github.com/nidhibaid)),