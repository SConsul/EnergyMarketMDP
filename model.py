import torch 
import torch.nn as nn
import torch.nn.functional as F

class q_network(nn.Module):
    def __init__(self, state_size,action_size, seed, hidden_unit1=64,
                 hidden_unit2 = 64):
        """ Initialize parameters and build model. """
        super(q_network,self).__init__() 
        self.seed = torch.manual_seed(seed)
        self.layer1= nn.Linear(state_size,hidden_unit1)
        self.layer2 = nn.Linear(hidden_unit1,hidden_unit2)
        self.layer3 = nn.Linear(hidden_unit2,action_size)
        
    def forward(self,x):
        """ Build a network that maps state to action values. """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)