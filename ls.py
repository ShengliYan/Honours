import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
X = data['federal_funds'].iloc[0:23051]
X = X.to_numpy()
X = np.flip(X)


def transformations(X, delta_t):

    delta_X = np.diff(X)
    sqrt_X_ti = np.sqrt(X[:-1])
    
    # Calculate y_i, x1_i, x2_i
    y_i = delta_X / sqrt_X_ti
    x1_i = 1 / (delta_t * sqrt_X_ti)
    x2_i = sqrt_X_ti * delta_t
    
    # Return as matrices/vectors
    return y_i, np.vstack((x1_i, x2_i)).T


def objective(beta, X, y):
    beta = np.reshape(beta, (2, -1))
    return np.sum((y - X.dot(beta))**2)


delta_t = 1/23051  
t = np.linspace(0, 1, 23051)

y, X_transformed = transformations(X, delta_t)


initial_beta = np.array([0.1, 0.1])

result = minimize(objective, initial_beta, args=(X_transformed, y), method='L-BFGS-B')

beta_hat = result.x
beta1_hat, beta2_hat = beta_hat

alpha_hat = -beta2_hat
mu_hat = beta1_hat / alpha_hat
residuals = y - X_transformed.dot(beta_hat)
sigma_hat_squared_delta_t = np.sum(residuals**2) / len(y)
sigma_hat = np.sqrt(sigma_hat_squared_delta_t / delta_t)

print(f"alpha_hat:{alpha_hat:.8g}, mu_hat:{mu_hat:.8g}, sigma_hat:{sigma_hat:.8g}")


plt.plot(t, X)

plt.title(f'Data')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.show()

# alpha_hat:-0.1000038857109704, mu_hat:-1.129300054870608e-06, sigma_hat:26.332670952158626