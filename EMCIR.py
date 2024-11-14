import numpy as np
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
# Parameters
alpha = 0.01
mu = 5
sigma = 0.01
initial_condition = 1.13
T = 100
N = 10000
dt = T / N
num_paths = 1

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
            X[i + 1] = 0 
    X_paths[j] = X

# Calculate mean and variance
mean_across_paths = np.mean(X_paths, axis=0)
variance_across_paths = np.var(X_paths, axis=0)

mean = initial_condition * np.exp(-alpha * T) + mu*(1 - np.exp(-alpha*T))
variance = initial_condition * (sigma * sigma / (alpha))*(np.exp(-alpha*T)-np.exp(-2*alpha*T)) + (mu * sigma * sigma / (2*alpha))*(1-np.exp(-alpha*T))*(1-np.exp(-alpha*T))

print(f"Mean of Vasicek model across {num_paths} paths:", mean_across_paths[-1])
print(f"Variance of Vasicek model across {num_paths} paths:", variance_across_paths[-1])
print(f"Mean: {mean}")
print(f"Variance: {variance}")
# Plotting some sample paths
t = np.linspace(0, T, N + 1)
for j in range(num_paths):
    plt.plot(t, X_paths[j], label='Path {}'.format(j+1))

plt.title(f'Simulation of CIR Model for a = {alpha} ')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.grid(True)

folder_path = "Plot/CIR" 
plt.savefig(f"{folder_path}/a = {alpha}.png", dpi=300)
plt.show()
