import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1
mu = 0.05
sigma = 0.1
initial_condition = 0.03
T = 1.0  
N = 1000  
dt = T / N  

# Wiener process increments
dW = np.random.normal(0, np.sqrt(dt), N)

t_values = np.linspace(0, T, N+1)
X_values = np.zeros(N+1)

X_values[0] = initial_condition

# Euler-Maruyama method
for i in range(N):
    drift = alpha * (mu - X_values[i])
    diffusion = sigma
    X_values[i+1] = X_values[i] + drift * dt + diffusion * dW[i]

plt.plot(t_values, X_values, label='Vasicek Model')
plt.title('Simulation of Vasicek Model using Euler-Maruyama Method')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.legend()
plt.show()
