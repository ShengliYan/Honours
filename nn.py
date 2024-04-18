import numpy as np
from scipy.optimize import least_squares, minimize
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load data from CSV
data = pd.read_csv('data.csv')

X = data['federal_funds'].iloc[0:23051]
X = X.to_numpy()
X = np.flip(X)
print(X)


def objective_ls(params, X, dt):
    b1, b2 = params
    y = (np.roll(X, -1) - X) / np.sqrt(X)
    x1 = dt / np.sqrt(X)
    x2 = np.sqrt(X) * dt
    epsilon = sigma * np.random.randn(len(X))
    model = b1 * x1 + b2 * x2 + epsilon
    return y - model


total_days = 23051
dt = 1.0 / total_days
sigma = 0.02  # Example value for volatility parameter
initial_guess = [0.2, -0.6]  # Initial guess for b1 and b2

result_ls = least_squares(objective_ls, initial_guess, args=(X, dt))
b1_optimal_ls, b2_optimal_ls = result_ls.x
print("Optimal values for b1 and b2 using least_squares:", b1_optimal_ls, b2_optimal_ls)
