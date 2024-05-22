import torch
import torch.optim as optim
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv('data.csv')

X = data['federal_funds'].iloc[0:23051].to_numpy()
X = np.flip(X)
X = X.copy()

# Normalize the data
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

observed_data = torch.tensor(X_normalized, dtype=torch.float32).to(device)

time_steps = torch.arange(len(observed_data), dtype=torch.float32).to(device)
delta_t = 1

alpha = torch.tensor(0.1, requires_grad=True, device=device)
mu = torch.tensor(0.1, requires_grad=True, device=device)
sigma = torch.tensor(0.1, requires_grad=True, device=device)

def loss_function(predicted, actual):
    return torch.mean((predicted - actual) ** 2)

optimizer = optim.Adam([alpha, mu, sigma], lr=0.001)

epochs = 100000
epsilon = 1e-6

for epoch in range(epochs):
    optimizer.zero_grad()
    
    y_pred = torch.zeros_like(observed_data)
    y_pred[0] = observed_data[0]
    noise = torch.randn(len(time_steps) - 1, device=device)
    
    y_pred[1:] = y_pred[:-1] + alpha * (mu - y_pred[:-1]) * delta_t + sigma * torch.sqrt(torch.relu(y_pred[:-1]) + epsilon) * noise
    
    loss = loss_function(y_pred, observed_data)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_([alpha, mu, sigma], max_norm=1)
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, alpha: {alpha.item()}, mu: {mu.item()}, sigma: {sigma.item()}')

# De-normalize the parameters
alpha_de_normalized = alpha.item() / X_std
mu_de_normalized = mu.item() * X_std + X_mean
sigma_de_normalized = sigma.item() * np.sqrt(X_std)

print(f'De-normalized parameter alpha: {alpha_de_normalized}, mu: {mu_de_normalized}, sigma: {sigma_de_normalized}')
#De-normalized parameter alpha: 0.008767448711324647, mu: 4.893875683463864, sigma: -0.06480797620918553