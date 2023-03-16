class DER(object):
    def __init__(self,power_capacity,energy_capacity,energy_demand):
        self.power_cap = power_capacity
        self.energy_cap = energy_capacity
        self.energy_dem = energy_demand
    
    def nb_states(self):
        '''The state of the system can go from 0 (completely
        discharged) to energy_cap (fully charged)'''
        return self.energy_cap+1
    
    def nb_actions(self):
        '''An action is to charge or discharge the system up to
        power_cap (fastest rate at which we can charge). So, the
        actions are within [-power_cap, power_cap]'''
        return 2*self.power_cap+1
