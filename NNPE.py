import torch
import torch.optim as optim
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv('data.csv')

X = data['federal_funds'].iloc[13051:23051].to_numpy()
X = np.flip(X)
X = X.copy()

observed_data = torch.tensor(X, dtype=torch.float32).to(device)

time_steps = torch.arange(len(observed_data), dtype=torch.float32).to(device)
delta_t = 40/10000

alpha = torch.tensor(0.1, requires_grad=True, device=device)
mu = torch.tensor(0.1, requires_grad=True, device=device)
sigma = torch.tensor(0.0, requires_grad=True, device=device)

def loss_function(predicted, actual):
    return torch.mean((predicted - actual) ** 2)

optimizer = optim.Adam([alpha, mu, sigma], lr=0.01)

epochs = 5000
epsilon = 1e-6

for epoch in range(epochs):
    optimizer.zero_grad()
    
    y_pred = torch.zeros_like(observed_data)
    y_pred[0] = observed_data[0]
    noise = torch.randn(len(time_steps) - 1, device=device)
    scaled_noise = noise * torch.sqrt(torch.tensor(delta_t))
    
    y_pred[1:] = y_pred[:-1] + alpha * (mu - y_pred[:-1]) * delta_t
    
    loss = loss_function(y_pred, observed_data)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_([alpha, mu, sigma], max_norm=1)
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, alpha: {alpha.item()}, mu: {mu.item()}, sigma: {sigma.item()}')

print(f'Estimated parameter alpha: {alpha.item()}, mu: {mu.item()}, sigma: {sigma.item()}')
