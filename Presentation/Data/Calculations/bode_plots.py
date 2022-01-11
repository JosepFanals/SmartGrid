import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n2 = 0
n1 = 0
n0 = 100
d2 = 1
d1 = 14.142
d0 = 100

# low
f = np.arange(0.1, 100, 0.1)
mag = []
ang = []
for ff in f:
    w = 2 * np.pi * ff
    mag_num = np.sqrt((n2 * (- w ** 2) + n0) ** 2 + (w * n1) ** 2)
    mag_den = np.sqrt((d2 * (- w ** 2) + d0) ** 2 + (w * d1) ** 2)
    mag.append(20 * np.log10 (mag_num / mag_den))

    ang_num = np.arctan2((w * n1) , (n2 * (- w ** 2) + n0))
    ang_den = np.arctan2((w * d1) , (d2 * (- w ** 2) + d0))
    ang.append((ang_num - ang_den) * 180 / np.pi)



# plt.plot(np.log10(f), ang)
# plt.show()

df_mag_low = pd.DataFrame(mag, f)
df_mag_low.to_csv('Data/Calculations/mag_low.csv')


df_ang_low = pd.DataFrame(ang, f)
df_ang_low.to_csv('Data/Calculations/ang_low.csv')


# high
n2 = 1
n1 = 0
n0 = 0
d2 = 1
d1 = 14.142
d0 = 100

f = np.arange(0.1, 100, 0.1)
mag = []
ang = []
for ff in f:
    w = 2 * np.pi * ff
    mag_num = np.sqrt((n2 * (- w ** 2) + n0) ** 2 + (w * n1) ** 2)
    mag_den = np.sqrt((d2 * (- w ** 2) + d0) ** 2 + (w * d1) ** 2)
    mag.append(20 * np.log10 (mag_num / mag_den))

    ang_num = np.arctan2((w * n1) , (n2 * (- w ** 2) + n0))
    ang_den = np.arctan2((w * d1) , (d2 * (- w ** 2) + d0))
    ang.append((ang_num - ang_den) * 180 / np.pi)



# plt.plot(np.log10(f), ang)
# plt.show()

df_mag_high = pd.DataFrame(mag, f)
df_mag_high.to_csv('Data/Calculations/mag_high.csv')


df_ang_high = pd.DataFrame(ang, f)
df_ang_high.to_csv('Data/Calculations/ang_high.csv')


