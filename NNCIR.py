import torch
import torchsde
import numpy as np
import matplotlib.pyplot as plt
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.setrecursionlimit(1500)
torch.manual_seed(42)

# Parameters for cir model
alpha = 1
mu = 5
sigma = 0.01
initial_condition = 1.13
T = 100

batch_size, state_size, brownian_size = 1, 1, 1
t_size = 10000

class CIRSDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()

    # Drift function for cir model
    def f(self, t, y):
        return alpha * (mu - y)  # shape (batch_size, state_size)

    # Diffusion function for Vasicek model
    def g(self, t, y):
        return sigma * np.sqrt(y)

sde_cir = CIRSDE()
y0_cir = torch.full((batch_size, state_size), initial_condition)  # Starting at the mean value
ts_cir = torch.linspace(0, T, t_size)  # Adjust time interval as needed

# Solve the SDE using torchsde.sdeint
ys_cir = torchsde.sdeint(sde_cir, y0_cir, ts_cir, method='euler')

# Extracting time and state values for plotting
t_values_cir = ts_cir.detach().numpy()  # Convert tensor to NumPy array
y_values_cir = ys_cir.squeeze(2).detach().numpy()  # Squeeze to remove extra dimension

# Calculate mean and variance of paths
mean_values_cir = np.mean(y_values_cir, axis=1)
variance_values_cir = np.var(y_values_cir, axis=1)

mean = initial_condition * np.exp(-alpha * T) + mu*(1 - np.exp(-alpha*T))
variance = initial_condition * (sigma * sigma / (alpha))*(np.exp(-alpha*T)-np.exp(-2*alpha*T)) + (mu * sigma * sigma / (2*alpha))*(1-np.exp(-alpha*T))*(1-np.exp(-alpha*T))
# Print mean and variance
print(f"Mean values across paths:\n{mean_values_cir[-1]}")
print(f"Variance values across paths:\n{variance_values_cir[-1]}")
print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plotting y(t) against t for cir model
plt.figure(figsize=(8, 6))
for i in range(batch_size):
    plt.plot(t_values_cir, y_values_cir[:, i], label=f'Path {i+1}')

plt.title(f'Simulation of CIR Model by neural networks for a = {alpha}')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.grid(True)

folder_path = "Plot/NNCIR" 
plt.savefig(f"{folder_path}/a = {alpha}.png", dpi=300)
plt.show()
