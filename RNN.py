import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv('data.csv')

X = data['federal_funds'].iloc[0:23051].to_numpy()
X = np.flip(X)
X = X.copy()

# Shift data to be positive before normalization
X_shift = np.abs(np.min(X)) + 1
X += X_shift

# Normalize the data
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

observed_data = torch.tensor(X_normalized, dtype=torch.float32).to(device)

# Create sequences of data
sequence_length = 10  # Example sequence length

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_normalized, sequence_length)
X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_seq = torch.tensor(y_seq, dtype=torch.float32).to(device)

# Reshape data
X_seq = X_seq.unsqueeze(-1)
y_seq = y_seq.unsqueeze(-1)

# Define the RNN Model
class CIRRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CIRRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1  # Number of input features
hidden_dim = 50  # Number of hidden units
output_dim = 3  # Number of output features (alpha, mu, sigma)
num_layers = 2  # Number of RNN layers

model = CIRRNN(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Gradient clipping value
clip_value = 1.0

# Training the RNN model
epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_seq)
    alpha_pred, mu_pred, sigma_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    
    # Ensure sigma_pred is positive by taking the absolute value
    sigma_pred = torch.abs(sigma_pred)
    
    # Compute the CIR process output based on predicted parameters
    X_t = X_seq[:, -1, 0]  # Last value in each sequence
    
    # Ensure non-negativity
    X_t = torch.relu(X_t + 1e-6)
    
    epsilon = torch.randn_like(X_t)  # Random noise
    
    X_t1_pred = X_t + alpha_pred * (mu_pred - X_t) * 1 + sigma_pred * torch.sqrt(X_t) * epsilon
    
    # Debugging intermediate values
    if torch.isnan(X_t1_pred).any():
        print("NaN detected in X_t1_pred")
        print("alpha_pred:", alpha_pred)
        print("mu_pred:", mu_pred)
        print("sigma_pred:", sigma_pred)
        print("X_t:", X_t)
        print("epsilon:", epsilon)
        break
    
    loss = criterion(X_t1_pred, y_seq.squeeze())
    
    # Debugging intermediate values
    if torch.isnan(loss).any():
        print("NaN detected in loss")
        print("X_t1_pred:", X_t1_pred)
        print("y_seq:", y_seq)
        break
    
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Extract the parameters from the trained model
model.eval()
with torch.no_grad():
    predicted_params = model(X_seq)
    alpha_pred, mu_pred, sigma_pred = predicted_params[:, 0], predicted_params[:, 1], predicted_params[:, 2]

# De-normalize the parameters
alpha_de_normalized = alpha_pred.cpu().numpy() / X_std
mu_de_normalized = mu_pred.cpu().numpy() * X_std + X_mean
sigma_de_normalized = sigma_pred.cpu().numpy() * np.sqrt(X_std)

print(f'De-normalized parameter alpha: {alpha_de_normalized.mean()}, mu: {mu_de_normalized.mean()}, sigma: {sigma_de_normalized.mean()}')
