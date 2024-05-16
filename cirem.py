
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Parameters
alpha = 0.0016554113
mu = 4.884768
sigma = 0.17352199
initial_condition = 1.13
T = 23051
N = 23051
dt = T / N
num_paths = 1000

# Array to store paths
X_paths = np.zeros((num_paths, N+1))

# Wiener process increments
dW_all = np.random.normal(0, np.sqrt(dt), (num_paths, N))

# Generate paths
for j in range(num_paths):
    dW = dW_all[j]
    X = np.zeros(N + 1)
    X[0] = initial_condition
    for i in range(N):
        drift = alpha * (mu - X[i])
        diffusion = sigma * np.sqrt(X[i])
        X[i + 1] = X[i] + drift * dt + diffusion * dW[i]
        if X[i + 1] < 0:
            X[i + 1] = 0  # Ensure non-negative values
    X_paths[j] = X

# Calculate mean and variance
mean_across_paths = np.mean(X_paths, axis=0)
variance_across_paths = np.var(X_paths, axis=0)

print(f"Mean of CIR model across {num_paths} paths:", mean_across_paths[-1])
print(f"Variance of CIR model across {num_paths} paths:", variance_across_paths[-1])

# Plotting some sample paths
t = np.linspace(0, T, N + 1)
for j in range(num_paths):
    plt.plot(t, X_paths[j], label='Path {}'.format(j+1))

plt.title(f'Simulation of Cox-Ingersoll-Ross Model ({num_paths} Paths)')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.show()

