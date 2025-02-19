# LSTM Stock Movement Binary Classifier

This repository contains a PyTorch-based LSTM model that predicts stock price movements—up or down—by analyzing historical stock data. The model leverages Yahoo Finance data and performs end-to-end processing from data acquisition to training, evaluation, and prediction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Processing](#data-processing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)

## Overview

This project implements an LSTM-based binary classifier to forecast whether a stock's closing price will increase or decrease on the following day. By processing sequences of historical data, the model generates a binary prediction output:
- **1:** Indicates an expected increase in the closing price.
- **0:** Indicates an expected decrease in the closing price.

## Features

- **Data Acquisition:** Downloads historical stock data using `yfinance`.
- **Data Preprocessing:** Normalizes and structures data for LSTM input.
- **LSTM Model:** Implements a multi-layer LSTM network with dropout regularization.
- **Training Pipeline:** Uses PyTorch DataLoader, AdamW optimizer, and a learning rate scheduler.
- **Evaluation Metrics:** Evaluates model performance using RMSE and R² scores.
- **Binary Prediction:** Outputs a binary signal based on the predicted percentage change.

## Installation

Ensure you have Python 3.x installed, then install the required dependencies:

```bash
pip install numpy pandas torch yfinance scikit-learn matplotlib
```

## Usage

To train and test the LSTM model, follow these steps:

```python
python train.py --ticker JPM --epochs 50 --batch_size 64
```

## Model Architecture

The LSTM model consists of:
- **LSTM Layers:** Two stacked LSTM layers for sequential data processing.
- **Dropout:** Applied to prevent overfitting.
- **Fully Connected Layer:** A linear layer to output the final prediction.

### Detailed Model Architecture

```python
import torch.nn as nn

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(LSTMStockPredictor, self).__init__()
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, 1)
        
        # Activation Function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM Forward Pass
        out, _ = self.lstm(x)
        
        # Fully Connected Layer
        out = self.fc(out[:, -1, :])
        
        # Sigmoid Activation for Binary Classification
        return self.sigmoid(out)
```

## Data Processing

The dataset is obtained from Yahoo Finance and normalized using `MinMaxScaler`. The preprocessing steps involve:
- **Feature Selection:** Using 'Open', 'High', 'Low', 'Close', 'Volume', and an average price feature.
- **Normalization:** Scaling features between 0 and 1 for efficient training.
- **Time Series Windowing:** Creating time-step sequences for LSTM input.

### Data Processing Code:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Avg'] = (df['High'] + df['Low']) / 2  
    return df

def prepare_data(df, time_steps=100):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    x_data, y_data = [], []
    for i in range(time_steps, len(scaled_data) - 1):
        x_data.append(scaled_data[i-time_steps:i])
        y_data.append(scaled_data[i+1, 3])
    return np.array(x_data), np.array(y_data), scaler
```

## Training and Evaluation

The model is trained using:
- **Optimizer:** AdamW with weight decay regularization.
- **Loss Function:** Mean Squared Error (MSE) Loss.
- **Learning Rate Scheduler:** Reduces learning rate upon stagnation.

### Training Code:

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=50):
    model.to(device)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return model
```

## Results

After training, the model evaluates performance using RMSE and R² score. A binary prediction is made based on the percentage change:

```python
predicted_change = ((predictions[-1] - actuals[-1]) / actuals[-1]) * 100
print("1" if predicted_change > 0 else "0")
```

## Future Work

- **Feature Engineering:** Adding technical indicators like RSI and MACD.
- **Hyperparameter Tuning:** Optimizing model parameters for better accuracy.
- **Alternative Architectures:** Testing CNN-LSTM hybrids or Transformer-based models.

