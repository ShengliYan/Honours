
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('data.csv')
X = data['federal_funds'].iloc[13051:23051]
X = X.to_numpy()
X = np.flip(X)
X_real = X



seed = 42
np.random.seed(seed)

# Parameters
alpha = 37.50943374633789
mu = 37.50909423828125
sigma = 0.35303235054016113
initial_condition = 1.13
T = 40
N = 9999
dt = T / N
num_paths = 1000

X_paths = np.zeros((num_paths, N+1))


dW_all = np.random.normal(0, np.sqrt(dt), (num_paths, N))


for j in range(num_paths):
    dW = dW_all[j]
    X = np.zeros(N + 1)
    X[0] = initial_condition
    for i in range(N):
        drift = alpha * (mu - X[i])
        diffusion = sigma * np.sqrt(X[i])
        X[i + 1] = X[i] + drift * dt + diffusion * dW[i]
        if X[i + 1] < 0:
            X[i + 1] = 0 
    X_paths[j] = X


mse = np.mean((X_paths - X_real) ** 2, axis=1)

min_mse_index = np.argmin(mse)
print(f"Path with minimal MSE: {min_mse_index+1}")
print(f"MSE: {mse[min_mse_index]}")


t = np.linspace(0, T, N + 1)
plt.plot(t, X_paths[min_mse_index], label='Best Path')
plt.plot(t, X_real, label='Real Data', linestyle='--')
plt.title(f'CIR Model Simulation (MSE = {mse[min_mse_index]})')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.legend()
plt.show()


'''

mean_across_paths = np.mean(X_paths, axis=0)
variance_across_paths = np.var(X_paths, axis=0)

print(f"Mean of CIR model across {num_paths} paths:", mean_across_paths[-1])
print(f"Variance of CIR model across {num_paths} paths:", variance_across_paths[-1])


t = np.linspace(0, T, N + 1)
for j in range(num_paths):
    plt.plot(t, X_paths[j], label='Path {}'.format(j+1))

plt.title(f'Simulation of {num_paths} paths')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.show()
'''