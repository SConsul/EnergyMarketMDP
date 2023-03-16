market1 = Market(1)
der1 = DER(power_capacity = 100 ,energy_capacity = 400, energy_demand = 100, h_demand=18)

agent1 = Agent(nb_states=der1.nb_states(),nb_actions=der1.nb_actions(),
               demand=der1.energy_dem,h_demand=der1.h_demand)
list_hour, list_price, list_power_supplied, list_actions, list_s = agent1.Q_learning(24,market1,der1,1)

fig, ax1 = plt.subplots()

ax1.set_xlabel('time (h)')
ax1.set_ylabel('price', color='r')
ax1.plot(list_hour,list_price, color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
#ax2.set_ylabel('sin', color=color)
ax2.plot(list_hour,list_power_supplied,label='solar power')
ax2.plot(list_hour,list_actions,label = 'actions')
ax2.plot(list_hour,list_s, label = 'charging state')

fig.tight_layout()
plt.legend()
plt.show()
