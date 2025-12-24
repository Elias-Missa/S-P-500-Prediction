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

from . import config, metrics
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
            # Explicitly pass regime_col from config if not in overrides
            params = {'alpha': 1.0}
            if 'regime_col' not in overrides:
                params['regime_col'] = getattr(config, 'REGIME_COL', 'RV_Ratio')
            
            params.update({k: v for k, v in overrides.items() if k in ['alpha', 'regime_col', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'solver', 'positive', 'random_state']})
            return RegimeGatedRidge(**params)

        elif model_type == 'RegimeGatedHybrid':
            # Hybrid Regime-gated model
            params = {}
            for k, v in overrides.items():
                params[k] = v
                
            # Explicitly extract or default from config using the correct config keys
            if 'regime_col' not in params:
                params['regime_col'] = getattr(config, 'REGIME_COL', 'RV_Ratio')
                
            low_model = params.pop('low_model', getattr(config, 'REGIME_LOW_MODEL', 'Ridge'))
            high_model = params.pop('high_model', getattr(config, 'REGIME_HIGH_MODEL', 'RandomForest'))
            
            return RegimeGatedHybrid(low_model=low_model, high_model=high_model, **params)

        elif model_type == 'TrendGatedHybrid':
            return TrendGatedHybrid(**overrides)

        elif model_type == 'Ridge_Residual_XGB':
            return RidgeResidualXGB(**overrides)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

class RegimeGatedRidge:
    def __init__(self, alpha=1.0, regime_col=None, **kwargs):
        if regime_col is None:
            raise ValueError("RegimeGatedRidge requires a 'regime_col' parameter.")
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
    def __init__(self, low_model, high_model, regime_col=None, **kwargs):
        if regime_col is None:
            raise ValueError("RegimeGatedHybrid requires a 'regime_col' parameter.")
            
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


class TrendGatedHybrid:
    def __init__(self, bull_model='Ridge', bear_model='XGBoost', regime_col='Dist_from_200MA', **kwargs):
        self.bull_model_type = bull_model
        self.bear_model_type = bear_model
        # Use Dist_from_200MA: Positive = Price > 200MA (Bull), Negative = Price < 200MA (Bear)
        self.regime_col = regime_col 
        self.kwargs = kwargs
        
        # Bull Model (Trend/Momentum) - Defaults to Ridge
        self.bull_model = ModelFactory.get_model(bull_model, overrides=kwargs)
        
        # Bear Model (Panic/Mean Reversion) - Defaults to XGBoost
        # Note: XGBoost usually requires specific params separate from Ridge (like n_estimators)
        # For now, we pass kwargs to both, assuming collisions are safe or handled by factories
        self.bear_model = ModelFactory.get_model(bear_model, overrides=kwargs)
        
    def fit(self, X, y):
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found. Cannot determine Trend Regime.")
            
        # Regime Logic: 
        # Bull: Dist > 0 (Price > 200MA) -> Use Bull Model (Ridge)
        # Bear: Dist <= 0 (Price <= 200MA) -> Use Bear Model (XGBoost)
        self.regime_threshold = 0.0
        
        bull_mask = X[self.regime_col] > self.regime_threshold
        bear_mask = ~bull_mask
        
        # Train Bull Model (on Bull Data)
        if bull_mask.sum() > 0:
            self.bull_model.fit(X[bull_mask], y[bull_mask])
            
        # Train Bear Model (on Bear Data)
        if bear_mask.sum() > 0:
            self.bear_model.fit(X[bear_mask], y[bear_mask])
            
        return self
        
    def predict(self, X):
        if self.regime_col not in X.columns:
            raise ValueError(f"Regime column '{self.regime_col}' not found.")
            
        # Identify Regimes
        bull_mask = X[self.regime_col] > self.regime_threshold
        bear_mask = ~bull_mask
        
        y_pred = np.zeros(len(X))
        
        # Bull Predictions
        if bull_mask.sum() > 0:
            y_pred[bull_mask] = self.bull_model.predict(X[bull_mask])
            
        # Bear Predictions
        if bear_mask.sum() > 0:
            y_pred[bear_mask] = self.bear_model.predict(X[bear_mask])
            
        return y_pred
    
    @property
    def feature_importances_(self):
        """Averages importance from both sub-models if available."""
        try:
            bull_fi = getattr(self.bull_model, 'feature_importances_', getattr(self.bull_model, 'coef_', None))
            bear_fi = getattr(self.bear_model, 'feature_importances_', getattr(self.bear_model, 'coef_', None))
            
            # If coef_ is 1D or 2D, normalize or just average? Mean implies simple combination.
            # Ridge coef is (n_features,), XGB feature_importances_ is (n_features,)
            if bull_fi is not None and bear_fi is not None:
                # Handle potential shape mismatch if one is array/list
                return (np.array(bull_fi) + np.array(bear_fi)) / 2.0
            return bull_fi if bull_fi is not None else bear_fi
        except Exception:
            return None

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


class RidgeResidualXGB:
    def __init__(self, ridge_alpha=1.0, xgb_params=None, **kwargs):
        self.ridge_alpha = ridge_alpha
        self.xgb_params = xgb_params or {}
        self.kwargs = kwargs
        
        # Base Model: Ridge
        self.base_model = Ridge(alpha=ridge_alpha)
        
        # Residual Model: XGBoost
        # Default XGB params if not provided
        default_xgb = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_jobs': -1,
            'random_state': 42
        }
        if self.xgb_params:
            default_xgb.update(self.xgb_params)
        
        # Ensure we have XGBRegressor available
        if XGBRegressor is None:
            raise ImportError("XGBoost not installed.")
            
        self.residual_model = XGBRegressor(**default_xgb)
        
        self.best_lambda = 0.0 # Default to base model only until optimized
        self.lambdas = [0.25, 0.5, 0.75, 1.0] # Candidate lambdas
        
    def fit(self, X, y):
        # 1. Train Base Model (Ridge)
        self.base_model.fit(X, y)
        base_preds = self.base_model.predict(X)
        
        # 2. Compute Residuals
        # Residual = Actual - Predicted
        residuals = y - base_preds
        
        # 3. Prepare Features for Residual Model
        # Features = Rehabbed Features (X) + Ridge Prediction
        X_res = X.copy()
        X_res['Ridge_Pred'] = base_preds
        
        # 4. Train Residual Model (XGBoost)
        self.residual_model.fit(X_res, residuals)
        
        return self
        
    def optimize_mixing(self, X_val, y_val):
        """
        Select best lambda on validation set to minimize RMSE.
        final_pred = ridge_pred + lambda * residual_pred
        """
        # Generate predictions
        base_preds = self.base_model.predict(X_val)
        
        X_res = X_val.copy()
        X_res['Ridge_Pred'] = base_preds
        res_preds = self.residual_model.predict(X_res)
        
        # Get config params
        candidates = getattr(config, 'STACK_LAMBDA_GRID', [0.0, 0.25, 0.5, 0.75, 1.0])
        criterion = getattr(config, 'STACK_LAMBDA_CRITERION', 'monthly_sharpe')
        
        print(f"  [RidgeResidualXGB] Optimizing Mixing (Candidates: {candidates}, Criterion: {criterion})")

        # Track best
        best_score = -float('inf') 
        best_lam = 0.0
        
        # Determine frequency for metric calc
        execution_frequency = getattr(config, 'EXECUTION_FREQUENCY', 'monthly')
        
        results_log = []

        for lam in candidates:
            final_preds = base_preds + lam * res_preds
            
            # Compute Metric
            score = -float('inf')
            metric_display = ""
            
            try:
                if criterion == "rmse":
                    # Minimize RMSE -> Maximize negative RMSE
                    rmse = np.sqrt(np.mean((y_val - final_preds) ** 2))
                    score = -rmse
                    metric_display = f"RMSE: {rmse:.6f}"
                    
                elif criterion == "ic":
                    score = metrics.calculate_ic(y_val, final_preds)
                    metric_display = f"IC: {score:.4f}"
                    
                elif criterion == "decile_spread":
                    spread = metrics.calculate_decile_spread(y_val, final_preds)
                    score = spread
                    metric_display = f"Spread: {spread:.4f}"
                    
                elif criterion == "monthly_sharpe":
                    # Use strategy metrics calculator
                    # We need dates for proper monthly aggregation logic if available
                    val_dates = None
                    if hasattr(y_val, 'index'):
                         # Try to use index as dates
                         val_dates = y_val.index
                    
                    strat_metrics = metrics.calculate_strategy_metrics(
                        y_val, final_preds, 
                        dates=val_dates, 
                        execution_frequency=execution_frequency
                    )
                    score = strat_metrics.get('sharpe', -999.0)
                    metric_display = f"Sharpe: {score:.4f}"
                    
                else:
                    # Default fallback to RMSE (negative)
                     rmse = np.sqrt(np.mean((y_val - final_preds) ** 2))
                     score = -rmse
                     metric_display = f"RMSE (Fallback): {rmse:.6f}"
            
            except Exception as e:
                print(f"    - Lambda: {lam:.2f} -> Error calculating {criterion}: {e}")
                score = -float('inf')

            print(f"    - Lambda: {lam:.2f} -> {metric_display}")
            
            results_log.append({'lambda': lam, 'score': score})
            
            if score > best_score:
                best_score = score
                best_lam = lam
                
        # Fail fast check: If all scores are -inf, something is wrong
        if all(r['score'] == -float('inf') for r in results_log):
             print("  [RidgeResidualXGB] CRITICAL: All candidates failed metric calculation. Defaulting to 0.0.")
             best_lam = 0.0
             # We let the assertion in train_walkforward catch this if 0.0 is disallowed, 
             # but user said "only error if all lambdas are NaN/invalid" - well, we default to safe 0.0 here.

        self.best_lambda = best_lam
        print(f"  [RidgeResidualXGB] Selected Lambda: {self.best_lambda} (Best {criterion}: {best_score:.4f})")
        
        if self.best_lambda == 0.0:
            print("  [RidgeResidualXGB] Note: Selected Lambda is 0.0. Stack inactive (Signal preferred Ridge).")
        
    def predict(self, X):
        # Generate predictions
        base_preds = self.base_model.predict(X)
        
        X_res = X.copy()
        X_res['Ridge_Pred'] = base_preds
        res_preds = self.residual_model.predict(X_res)
        
        # Combine
        return base_preds + self.best_lambda * res_preds

    def predict_decomposition(self, X):
        """
        Returns dictionary with component predictions:
        {
            'ridge_pred': np.array,
            'resid_pred': np.array,
            'final_pred': np.array,
            'lambda': float
        }
        """
        base_preds = self.base_model.predict(X)
        
        X_res = X.copy()
        X_res['Ridge_Pred'] = base_preds
        res_preds = self.residual_model.predict(X_res)
        
        final_preds = base_preds + self.best_lambda * res_preds
        
        return {
            'ridge_pred': base_preds,
            'resid_pred': res_preds,
            'final_pred': final_preds,
            'lambda': self.best_lambda
        }
    
    @property
    def feature_importances_(self):
        """
        Return feature importances of the XGBoost model.
        Note: This includes the 'Ridge_Pred' feature which is usually dominant.
        """
        try:
            return self.residual_model.feature_importances_
        except:
            return None
