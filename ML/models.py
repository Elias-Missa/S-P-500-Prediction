from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

import torch
import torch.nn as nn

from . import config

class ModelFactory:
    @staticmethod
    def get_model(model_type):
        if model_type == 'LinearRegression':
            return LinearRegression()
            
        elif model_type == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                random_state=config.RF_RANDOM_STATE,
                n_jobs=-1
            )
            
        elif model_type == 'XGBoost':
            if XGBRegressor is None:
                raise ImportError("XGBoost not installed.")
            return XGBRegressor(
                n_estimators=config.XGB_N_ESTIMATORS,
                learning_rate=config.XGB_LEARNING_RATE,
                max_depth=config.XGB_MAX_DEPTH,
                n_jobs=-1,
                random_state=42
            )
            
        elif model_type == 'MLP':
            return MLPRegressor(
                hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
                learning_rate_init=config.MLP_LEARNING_RATE_INIT,
                alpha=config.MLP_ALPHA,
                max_iter=config.MLP_MAX_ITER,
                random_state=42,
                early_stopping=True
            )
            
        elif model_type == 'LSTM':
            return LSTMModel(
                input_dim=34, # Will be dynamic in training loop, but class needs init
                hidden_dim=config.LSTM_HIDDEN_DIM,
                num_layers=config.LSTM_LAYERS,
                output_dim=1
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
