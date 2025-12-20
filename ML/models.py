import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
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
from .tft_model import TFTModel

class ModelFactory:
    @staticmethod
    def get_model(model_type, input_dim=None, overrides=None):
        overrides = overrides or {}
        if model_type == 'LinearRegression':
            return LinearRegression()

        elif model_type == 'Ridge':
            # Ridge regression with optional overrides (e.g., alpha)
            params = {'alpha': 1.0, 'random_state': 42}
            # Allow alpha and other sklearn Ridge kwargs via overrides
            params.update({k: v for k, v in overrides.items() if k in ['alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'solver', 'positive', 'random_state']})
            return Ridge(**params)

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
            
        elif model_type == 'TFT':
            if input_dim is None:
                raise ValueError("input_dim must be provided for TFT")
            return TFTModel(
                input_dim=input_dim,
                hidden_dim=config.TFT_HIDDEN_DIM,
                num_heads=config.TFT_NUM_HEADS,
                num_layers=config.TFT_LAYERS,
                dropout=config.TFT_DROPOUT,
                output_dim=1
            )

        elif model_type == 'RegimeGatedRidge':
            # Regime-gated Ridge regression
            params = {'alpha': 1.0, 'regime_col': 'RV_Ratio'}
            params.update({k: v for k, v in overrides.items() if k in ['alpha', 'regime_col', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'solver', 'positive', 'random_state']})
            return RegimeGatedRidge(**params)

        elif model_type == 'RegimeGatedHybrid':
            # Hybrid Regime-gated model (e.g. Ridge + RF)
            params = {}
            for k, v in overrides.items():
                params[k] = v
            # Extract sub-model types from config if not in overrides
            low_model = params.pop('low_model', getattr(config, 'REGIME_LOW_MODEL', 'Ridge'))
            high_model = params.pop('high_model', getattr(config, 'REGIME_HIGH_MODEL', 'RandomForest'))
            return RegimeGatedHybrid(low_model=low_model, high_model=high_model, **params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

class RegimeGatedRidge:
    def __init__(self, alpha=1.0, regime_col='RV_Ratio', **kwargs):
        self.alpha = alpha
        self.regime_col = regime_col
        self.kwargs = kwargs
        self.low_vol_model = Ridge(alpha=alpha, **kwargs)
        self.high_vol_model = Ridge(alpha=alpha, **kwargs)
        self.regime_threshold = None
        
    def fit(self, X, y):
        # Check if regime_col is in X
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found in features.")
            
        # Calculate median of regime_col
        self.regime_threshold = X[self.regime_col].median()
        
        # Split data
        low_vol_mask = X[self.regime_col] <= self.regime_threshold
        high_vol_mask = ~low_vol_mask
        
        # Train models
        if low_vol_mask.sum() > 0:
            self.low_vol_model.fit(X[low_vol_mask], y[low_vol_mask])
        
        if high_vol_mask.sum() > 0:
            self.high_vol_model.fit(X[high_vol_mask], y[high_vol_mask])
            
        return self

    @property
    def feature_importances_(self):
        """Returns average coefficients of low and high vol models."""
        try:
            return (self.low_vol_model.coef_ + self.high_vol_model.coef_) / 2.0
        except Exception:
            return None
        
        
    def predict(self, X):
        if self.regime_threshold is None:
            raise ValueError("Model not fitted yet.")
            
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found in features.")
            
        # Identify regimes
        low_vol_mask = X[self.regime_col] <= self.regime_threshold
        high_vol_mask = ~low_vol_mask
        
        # Predict
        y_pred = np.zeros(len(X))
        
        if low_vol_mask.sum() > 0:
            y_pred[low_vol_mask] = self.low_vol_model.predict(X[low_vol_mask])
            
        if high_vol_mask.sum() > 0:
            y_pred[high_vol_mask] = self.high_vol_model.predict(X[high_vol_mask])
            
        return y_pred

class RegimeGatedHybrid:
    def __init__(self, low_model='Ridge', high_model='RandomForest', regime_col='RV_Ratio', **kwargs):
        self.low_model_type = low_model
        self.high_model_type = high_model
        self.regime_col = regime_col
        self.kwargs = kwargs
        
        # Instantiate sub-models using ModelFactory
        # We need to handle potential recursion if we passed RegimeGatedHybrid again, but we won't do that.
        # We also need to pass kwargs down, but ModelFactory.get_model takes overrides.
        
        # Note: ModelFactory is a static class, so we can call it directly.
        # However, we are inside models.py, so we can just call ModelFactory.get_model.
        # But wait, ModelFactory is defined above.
        
        self.low_vol_model = ModelFactory.get_model(low_model, overrides=kwargs)
        self.high_vol_model = ModelFactory.get_model(high_model, overrides=kwargs)
        
        self.regime_threshold = None
        
    def fit(self, X, y):
        # Check if regime_col is in X
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found in features.")
            
        # Calculate median of regime_col
        self.regime_threshold = X[self.regime_col].median()
        
        # Split data
        low_vol_mask = X[self.regime_col] <= self.regime_threshold
        high_vol_mask = ~low_vol_mask
        
        # Train models
        if low_vol_mask.sum() > 0:
            self.low_vol_model.fit(X[low_vol_mask], y[low_vol_mask])
        
        if high_vol_mask.sum() > 0:
            self.high_vol_model.fit(X[high_vol_mask], y[high_vol_mask])
            
        return self

    @property
    def feature_importances_(self):
        """Averages importance from both sub-models."""
        try:
            low_fi = getattr(self.low_vol_model, 'feature_importances_', getattr(self.low_vol_model, 'coef_', None))
            high_fi = getattr(self.high_vol_model, 'feature_importances_', getattr(self.high_vol_model, 'coef_', None))
            if low_fi is not None and high_fi is not None:
                return (low_fi + high_fi) / 2.0
            return low_fi if low_fi is not None else high_fi
        except Exception:
            return None
        
        
    def predict(self, X):
        if self.regime_threshold is None:
            raise ValueError("Model not fitted yet.")
            
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found in features.")
            
        # Identify regimes
        low_vol_mask = X[self.regime_col] <= self.regime_threshold
        high_vol_mask = ~low_vol_mask
        
        # Predict
        y_pred = np.zeros(len(X))
        
        if low_vol_mask.sum() > 0:
            y_pred[low_vol_mask] = self.low_vol_model.predict(X[low_vol_mask])
            
        if high_vol_mask.sum() > 0:
            y_pred[high_vol_mask] = self.high_vol_model.predict(X[high_vol_mask])
            
        return y_pred

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
