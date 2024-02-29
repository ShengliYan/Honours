import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1
mu = 0.0
sigma = 0.1
initial_condition = 0.03
T = 1.0  
N = 1000  
dt = T / N  

# Wiener process increments
dW = np.random.normal(0, np.sqrt(dt), N)

t = np.linspace(0, T, N+1)
X = np.zeros(N+1)

X[0] = initial_condition

# Euler-Maruyama method
for i in range(N):
    drift = alpha * (mu - X[i])
    diffusion = sigma
    X[i+1] = X[i] + drift * dt + diffusion * dW[i]

plt.plot(t, X, label='Vasicek Model')
plt.title('Simulation of Vasicek Model using Euler-Maruyama Method')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.legend()
plt.show()
