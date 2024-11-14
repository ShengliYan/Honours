import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


np.random.seed(42)

a = 0.2083503007888794
m = 6.115521430969238
s = -0.0004010692937299609

data = pd.read_csv('data.csv')
#X = data['federal_funds'].iloc[13371:23051]
X = data['federal_funds'].iloc[0:23051]
X = X.to_numpy()
X = np.flip(X)
monthly_X = X[:(len(X) // 30) * 30].reshape(-1, 30).mean(axis=1)


t = np.linspace(0, 40, 23051)
monthly_t = np.linspace(0, 40, len(monthly_X))

mean = X[0]*np.exp(-a*t) + m*(1-np.exp(-a*t))

print(f"The mean of the path is {mean}.")

plt.plot(monthly_t, monthly_X)
plt.title(f'Data')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.show()