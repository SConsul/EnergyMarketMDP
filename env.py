import numpy as np
import pandas as pd
import os

class Market(object):
    def __init__(self,seed, num_panels=100):
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.cloudy = None
        self.num_panels = num_panels

    def init_day(self,episode):
        self.t = 0
        self.cloudy = self.rng.integers(3,10)*self.rng.binomial(1, 0.1) #assume day is cloudy 10% of the time
                                                                        #cloudy>0 ->cloudy (larger == more dense clouds)
        file_list = os.listdir("./data/data_demand")
        file_path = "./data/data_demand/"+file_list[episode]
        with open(file_path,'r') as f:
            f.readline()
            f.readline()
            line = f.readline()
            self.global_demand = line.split(',')[1:-1:12]

        # file_path = "./data/data_household/"+os.listdir("./data/data_household")[episode]
        # df = pd.read_csv(file_path)
        # hd = df[df.columns[1]].values.tolist()
        # scale = np.random.normal(30,2)/max(hd)
        # self.house_demand = [scale*float(x) for x in hd]

    def solar_power(self, hour):
        var = 4
        
        peak = 0.4*self.num_panels/(1+ self.cloudy) #on cloudy day, power can reduce to 10-25% normal (in kW), single panel is rated for 250-400W
        power_supplied = np.exp(-((hour-12)**2/(2*var))) - np.exp(-(64/(2*var))) + self.rng.normal(0,0.2)
        power_supplied = np.exp(-((hour-12)**2/(2*var))) - np.exp(-(64/(2*var))) + np.random.normal(0,0.2)
        power_supplied = max(0,peak*power_supplied)

        return power_supplied


    def price_from_duck_curve(self, hour): #price is $/kWh
        rand_num = self.rng.random()
        return 0.20 + 0.1*float(self.global_demand[hour])*(1+0.10*rand_num)/22000.0

    def step(self,episode):
        if self.t == 24 or self.cloudy is None:
            self.init_day(episode)

        hour = self.t % 24
        price = self.price_from_duck_curve(hour)
        power_supply = self.solar_power(hour)
        self.t +=1

        return price, power_supply
