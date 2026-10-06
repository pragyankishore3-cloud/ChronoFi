import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle

from model import ChronoFiLSTM

# Hyperparameters
SEQ_LENGTH = 12
EPOCHS = 150
LEARNING_RATE = 0.001
HIDDEN_SIZE = 50
NUM_LAYERS = 2
DROPOUT = 0.2

def fetch_and_prepare_data():
    print("Fetching historical S&P 500 data...")
    # Get S&P 500 data
    data = yf.download("^GSPC", start="1990-01-01", end="2024-01-01", progress=False)
    
    # Resample to monthly and get the close price
    monthly_data = data['Close'].resample('ME').last()
    
    # Calculate monthly returns
    returns = monthly_data.pct_change().dropna().values.reshape(-1, 1)
    
    # Scale returns to [-1, 1] for better LSTM convergence
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_returns = scaler.fit_transform(returns)
    
    # Save scaler for inference
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return scaled_returns

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    scaled_returns = fetch_and_prepare_data()
    
    X, y = create_sequences(scaled_returns, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    
    model = ChronoFiLSTM(
        input_size=1, 
        hidden_layer_size=HIDDEN_SIZE, 
        output_size=1, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        y_pred = model(X_train)
        
        single_loss = loss_function(y_pred, y_train)
        single_loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            print(f'Epoch {epoch:3} loss: {single_loss.item():10.8f}')
            
    print(f'Final Epoch {EPOCHS:3} loss: {single_loss.item():10.8f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'chronofi_lstm.pth')
    print("Model saved to chronofi_lstm.pth")

if __name__ == "__main__":
    train_model()
