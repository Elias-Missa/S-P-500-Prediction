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
from .transformer import TransformerModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, input_dim=None, overrides=None):
        overrides = overrides or {}
        if model_type == 'LinearRegression':
            return LinearRegression()

        elif model_type == 'RandomForest':
            params = {
                'n_estimators': config.RF_N_ESTIMATORS,
                'max_depth': config.RF_MAX_DEPTH,
                'min_samples_split': config.RF_MIN_SAMPLES_SPLIT,
                'min_samples_leaf': config.RF_MIN_SAMPLES_LEAF,
                'random_state': config.RF_RANDOM_STATE,
                'n_jobs': -1
            }
            params.update({k: v for k, v in overrides.items() if k in params})
            return RandomForestRegressor(**params)

        elif model_type == 'XGBoost':
            if XGBRegressor is None:
                raise ImportError("XGBoost not installed.")
            params = {
                'n_estimators': config.XGB_N_ESTIMATORS,
                'learning_rate': config.XGB_LEARNING_RATE,
                'max_depth': config.XGB_MAX_DEPTH,
                'n_jobs': -1,
                'random_state': 42
            }
            params.update({k: v for k, v in overrides.items() if k in params})
            return XGBRegressor(**params)

        elif model_type == 'MLP':
            params = {
                'hidden_layer_sizes': config.MLP_HIDDEN_LAYERS,
                'learning_rate_init': config.MLP_LEARNING_RATE_INIT,
                'alpha': config.MLP_ALPHA,
                'max_iter': config.MLP_MAX_ITER,
                'random_state': 42,
                'early_stopping': True
            }
            params.update({k: v for k, v in overrides.items() if k in params})
            return MLPRegressor(**params)

        elif model_type == 'LSTM':
            if input_dim is None:
                raise ValueError("input_dim must be provided for LSTM")
            return LSTMModel(
                input_dim=input_dim,
                hidden_dim=config.LSTM_HIDDEN_DIM,
                num_layers=config.LSTM_LAYERS,
                output_dim=1
            )
            
        elif model_type == 'CNN':
            if input_dim is None:
                raise ValueError("input_dim must be provided for CNN")
            return CNN1DModel(
                input_dim=input_dim,
                filters=config.CNN_FILTERS,
                kernel_size=config.CNN_KERNEL_SIZE,
                layers=config.CNN_LAYERS,
                dropout=config.CNN_DROPOUT,
                output_dim=1
            )
        
        elif model_type == 'Transformer':
            if input_dim is None:
                raise ValueError("input_dim must be provided for Transformer")
            return TransformerModel(
                input_dim=input_dim,
                model_dim=config.TRANSFORMER_MODEL_DIM,
                num_heads=config.TRANSFORMER_HEADS,
                num_layers=config.TRANSFORMER_LAYERS,
                dim_feedforward=config.TRANSFORMER_FEEDFORWARD_DIM,
                dropout=config.TRANSFORMER_DROPOUT
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

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, filters, kernel_size, layers, output_dim=1, dropout=0.2):
        super(CNN1DModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size, padding='same'))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        for _ in range(layers - 1):
            self.layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding='same'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Conv1d expects (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)
            
        # Global Average Pooling -> (batch, filters, 1)
        x = self.global_pool(x)
        
        # Flatten -> (batch, filters)
        x = x.squeeze(-1)
        
        # FC -> (batch, output_dim)
        out = self.fc(x)
        return out
