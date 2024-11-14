import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

C = 1.13  # X0
A = 3 # E(X)1
B = 20 # E(X)100

# Define the function
def func(alpha):
    term1 = C * np.exp(-40 * alpha)
    term2 = (A - C * np.exp(-alpha)) / (1 - np.exp(-alpha))
    term3 = 1 - np.exp(-40 * alpha)
    return term1 + term2 * term3 - B

# Initial guess
alpha_guess = 0.1

# Solve for alpha
alpha = fsolve(func, alpha_guess)
mu = (A - C*np.exp(-alpha))/(1 - np.exp(-alpha))

print("Solution for alpha:", alpha)
print("Solution for mu:", mu)

# Alpha: 0.23676464725220828, Mu: 10.000000000462766
#alpha = 0.13676464725220828
#mu = 10.000000000462766
t = np.linspace(0, 40, 23051)
X_t = C * np.exp(-alpha * t) + mu * (1 - np.exp(-alpha * t))

plt.figure(figsize=(10, 6))
plt.plot(t, X_t, label='X(t)')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title('Plot of X(t)')
plt.legend()
plt.grid(True)
plt.show()
