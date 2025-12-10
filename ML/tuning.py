import optuna
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from . import config, models, lstm_dataset
from .metrics import tail_weighted_mse, calculate_tail_metrics

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _f1_score(precision, recall):
    """Compute F1 score from precision and recall."""
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

class HyperparameterTuner:
    def __init__(self, model_type, X_train, y_train, X_val, y_val):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def optimize(self, n_trials=20):
        print(f"Starting Optuna optimization for {self.model_type} with {n_trials} trials...")
        
        study = optuna.create_study(direction="minimize")
        
        if self.model_type == 'LSTM':
            study.optimize(self._objective_lstm, n_trials=n_trials)
        elif self.model_type == 'XGBoost':
            study.optimize(self._objective_xgboost, n_trials=n_trials)
        elif self.model_type == 'RandomForest':
            study.optimize(self._objective_random_forest, n_trials=n_trials)
        elif self.model_type == 'MLP':
            study.optimize(self._objective_mlp, n_trials=n_trials)
        elif self.model_type == 'CNN':
            study.optimize(self._objective_cnn, n_trials=n_trials)
        else:
            print(f"Optuna not implemented for {self.model_type}. Skipping.")
            return {}
            
        print(f"Best params: {study.best_params}")
        print(f"Best objective (MSE - 0.5*F1): {study.best_value:.6f}")
        return study.best_params

    def _objective_lstm(self, trial):
        # Define Search Space
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Prepare Data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        time_steps = config.LSTM_TIME_STEPS
        X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, self.y_train.values, time_steps)
        X_val_seq, y_val_seq = lstm_dataset.create_sequences(X_val_scaled, self.y_val.values, time_steps)
        
        train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
        
        # Initialize Model
        input_dim = X_train_seq.shape[2]
        model = models.LSTMModel(input_dim, hidden_dim, num_layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Tail-weighted loss parameters
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail_alpha = getattr(config, 'TAIL_WEIGHT_ALPHA', 4.0)
        
        # Train Loop (Shortened for Tuning)
        epochs = 10 # Fewer epochs for tuning speed
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = tail_weighted_mse(outputs, y_batch, threshold=big_move_thresh, alpha=tail_alpha)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_pred = model(X_val_tensor).numpy().flatten()
            
        # Adjust actuals
        y_val_actual = self.y_val.iloc[time_steps-1:]
        
        # Compute MSE
        mse = mean_squared_error(y_val_actual, y_pred)
        
        # Compute big-move F1 scores
        tail = calculate_tail_metrics(y_val_actual.values, y_pred, threshold=big_move_thresh)
        big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
        big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
        
        # Combined objective: penalize MSE, reward big-move F1
        # Optuna minimizes, so subtract F1 terms
        objective = mse - 0.5 * (big_f1_up + big_f1_down)
        return objective

    def _objective_xgboost(self, trial):
        from xgboost import XGBRegressor
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        
        # Compute MSE
        mse = mean_squared_error(self.y_val, y_pred)
        
        # Compute big-move F1 scores
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail = calculate_tail_metrics(self.y_val.values, y_pred, threshold=big_move_thresh)
        big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
        big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
        
        # Combined objective: penalize MSE, reward big-move F1
        objective = mse - 0.5 * (big_f1_up + big_f1_down)
        return objective

    def _objective_random_forest(self, trial):
        from sklearn.ensemble import RandomForestRegressor
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        
        # Compute MSE
        mse = mean_squared_error(self.y_val, y_pred)
        
        # Compute big-move F1 scores
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail = calculate_tail_metrics(self.y_val.values, y_pred, threshold=big_move_thresh)
        big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
        big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
        
        # Combined objective: penalize MSE, reward big-move F1
        objective = mse - 0.5 * (big_f1_up + big_f1_down)
        return objective

    def _objective_mlp(self, trial):
        from sklearn.neural_network import MLPRegressor
        
        # Hidden layers: suggest number of layers and size
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 16, 128))
            
        params = {
            'hidden_layer_sizes': tuple(layers),
            'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-2),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-2),
            'max_iter': 200, # Lower for tuning
            'random_state': 42,
            'early_stopping': True
        }
        
        model = MLPRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        
        # Compute MSE
        mse = mean_squared_error(self.y_val, y_pred)
        
        # Compute big-move F1 scores
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail = calculate_tail_metrics(self.y_val.values, y_pred, threshold=big_move_thresh)
        big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
        big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
        
        # Combined objective: penalize MSE, reward big-move F1
        objective = mse - 0.5 * (big_f1_up + big_f1_down)
        return objective

    def _objective_cnn(self, trial):
        # Define Search Space
        filters = trial.suggest_categorical('filters', [32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        layers = trial.suggest_int('layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Prepare Data (Same as LSTM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        time_steps = config.LSTM_TIME_STEPS # Share time steps with LSTM for now
        X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, self.y_train.values, time_steps)
        X_val_seq, y_val_seq = lstm_dataset.create_sequences(X_val_scaled, self.y_val.values, time_steps)
        
        train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
        
        # Initialize Model
        input_dim = X_train_seq.shape[2]
        model = models.CNN1DModel(input_dim, filters, kernel_size, layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Tail-weighted loss parameters
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail_alpha = getattr(config, 'TAIL_WEIGHT_ALPHA', 4.0)
        
        # Train Loop
        epochs = 10
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = tail_weighted_mse(outputs, y_batch, threshold=big_move_thresh, alpha=tail_alpha)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_pred = model(X_val_tensor).numpy().flatten()
            
        y_val_actual = self.y_val.iloc[time_steps-1:]
        
        # Compute MSE
        mse = mean_squared_error(y_val_actual, y_pred)
        
        # Compute big-move F1 scores
        tail = calculate_tail_metrics(y_val_actual.values, y_pred, threshold=big_move_thresh)
        big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
        big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
        
        # Combined objective: penalize MSE, reward big-move F1
        # Optuna minimizes, so subtract F1 terms
        objective = mse - 0.5 * (big_f1_up + big_f1_down)
        return objective
