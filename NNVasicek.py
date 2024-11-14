import torch
import torchsde
import numpy as np
import matplotlib.pyplot as plt
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.setrecursionlimit(1500)
torch.manual_seed(42)

# Parameters for Vasicek model
alpha = 0.1
mu = 0.5
sigma = 0.01

batch_size, state_size, brownian_size = 1, 1, 1
t_size = 10000

class VasicekSDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()

    # Drift function for Vasicek model
    def f(self, t, y):
        # alpha = 0.1
        # mu = torch.exp(-t) + 5
        return alpha * (mu - y)  # shape (batch_size, state_size)

    # Diffusion function for Vasicek model
    def g(self, t, y):
        # sigma = 10*torch.exp(-t)
        return sigma * torch.ones_like(y)

sde_vasicek = VasicekSDE()
y0_vasicek = torch.full((batch_size, state_size), 1.13)  # Starting at the mean value
ts_vasicek = torch.linspace(0, 100, t_size)  # Adjust time interval as needed

# Solve the SDE using torchsde.sdeint
ys_vasicek = torchsde.sdeint(sde_vasicek, y0_vasicek, ts_vasicek, method='euler')

# Extracting time and state values for plotting
t_values_vasicek = ts_vasicek.detach().numpy()  # Convert tensor to NumPy array
y_values_vasicek = ys_vasicek.squeeze(2).detach().numpy()  # Squeeze to remove extra dimension

# Calculate mean and variance of paths
mean_values_vasicek = np.mean(y_values_vasicek, axis=1)
variance_values_vasicek = np.var(y_values_vasicek, axis=1)

mean = 1.13 * np.exp(-alpha * 100) + mu*(1 - np.exp(-alpha*100))
variance = (sigma * sigma / (2*alpha))*(1-np.exp(-2*alpha*100))
# Print mean and variance
print(f"Mean values across paths:\n{mean_values_vasicek[-1]}")
print(f"Variance values across paths:\n{variance_values_vasicek[-1]}")
print(f"Mean: {mean}")
print(f"Variance: {variance}")


# Plotting y(t) against t for Vasicek model
plt.figure(figsize=(8, 6))
for i in range(batch_size):
    plt.plot(t_values_vasicek, y_values_vasicek[:, i], label=f'Path {i+1}')

plt.title(f'Neural Network Simulation of Vasicek Model with u = {mu}')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.grid(True)

folder_path = "Plot/NNVasicek"  # Replace with your desired folder path
plt.savefig(f"{folder_path}/u = {mu}.png", dpi=300)
plt.show()
