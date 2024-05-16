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
    # print(np.vstack((x1_i, x2_i)).T)
    return y_i, np.vstack((x1_i, x2_i)).T


def objective_norm(beta, A, y):
    beta = np.reshape(beta, (2, -1))
    return norm(y - A @ (beta))**2

def matrix_solver(X, delta_t):
    delta_X = np.diff(X)
    sqrt_X_ti = np.sqrt(X[:-1])

    # Calculate y_i, x1_i, x2_i
    y_i = delta_X / sqrt_X_ti
    x1_i = 1 / (delta_t * sqrt_X_ti)
    x2_i = sqrt_X_ti * delta_t
    
    A = np.vstack((x1_i, x2_i)).T
    AT = np.vstack((x1_i, x2_i))
    
    ATA = AT @ A
    ATy = AT @ y_i
    
    beta_hat = np.linalg.solve(ATA, ATy)
    return beta_hat
    


delta_t = 1

beta_test = matrix_solver(X, delta_t)

print(f'beta_test by solving matrix: {beta_test}')


t = np.linspace(0, 23051, 23051)

y, X_transformed = transformations(X, delta_t)


initial_beta = np.array([0.1, 0.1])

result = minimize(objective_norm, initial_beta, args=(X_transformed, y), method='L-BFGS-B')

beta_hat = result.x
print(f'beta_hat by minimizing norm: {beta_hat}')
beta1_hat, beta2_hat = beta_test

alpha_hat = -beta2_hat
mu_hat = beta1_hat / alpha_hat
residuals = y - X_transformed.dot(beta_hat)
sigma_hat_squared_delta_t = np.sum(residuals**2) / len(y)
sigma_hat = np.sqrt(sigma_hat_squared_delta_t / delta_t)

print(f"alpha_hat:{alpha_hat:.8g}, mu_hat:{mu_hat:.8g}, sigma_hat:{sigma_hat:.8g}")
mean = X[0]*np.exp(-alpha_hat*23051) + mu_hat*(1-np.exp(-alpha_hat*23051))
variance = X[0]*(sigma_hat*sigma_hat/alpha_hat)*(np.exp(-alpha_hat*23051)-np.exp(-2*alpha_hat*23051)) + (mu_hat*sigma_hat*sigma_hat/(2*alpha_hat))*((1-np.exp(-alpha_hat*23051))*(1-np.exp(-alpha_hat*23051)))
print(f"The mean of the path is {mean}. The variance of the path is {variance}")

plt.plot(t, X)
plt.title(f'Data')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.show()

# beta_test by solving matrix: [ 0.0080863  -0.00165541]
# beta_hat by minimizing norm: [0.00154179 0.00150071]
# alpha_hat:-0.0015007102, mu_hat:-1.0273748, sigma_hat:0.17352194  By minimizing algorithm
# alpha_hat:0.0016554113, mu_hat:4.884768, sigma_hat:0.17352194 by solving matrix