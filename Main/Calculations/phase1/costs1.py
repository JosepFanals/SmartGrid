import numpy as np

# Data
Pvalley = 45  # euro/MWh
Pflat = 65  # euro/MWh
Ppeak = 90  # euro/MWh
Ppenalty = 180  # euro/MWh

tariff = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2]
# 1: valley, 2: flat, 3: peak

line_fail = 0.05  # fails/(km.year)
line_time = 2.5  # hours, adding disconnector??
trafo_fail = 0.15  # fails/year
trafo_time = 8  # incorrect the pdf?

# analyze a working day
P_slack = [-218.0051156, -202.4394667, -192.4183664, -188.2149818, -187.5097012, -192.3621666, -212.7323048, -249.171069, -274.1223931, -285.1850034, -289.3463065, -288.4510552, -285.5278164, -283.7341182, -273.3946683, -266.9860661, -263.2718686, -262.0091425, -267.6711039, -286.3870619, -298.4678664, -292.0277496, -267.3614921, -239.6611124] 

# Calculate
cost_slack = 0
for i in range(24):
    if tariff[i] == 1:
        cost_slack += Pvalley * P_slack[i]
    elif tariff[i] == 2:
        cost_slack += Pflat * P_slack[i]
    elif tariff[i] == 3:
        cost_slack += Ppeak * P_slack[i]
    else:
        print('error')

print('The cost of importing energy is: ', -cost_slack, 'euro')



