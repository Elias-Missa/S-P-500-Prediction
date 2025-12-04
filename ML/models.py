from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

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
                max_iter=config.MLP_MAX_ITER,
                random_state=42,
                early_stopping=True
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
