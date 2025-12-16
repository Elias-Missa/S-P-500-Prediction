import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from . import config, models, lstm_dataset, data_prep
from .metrics import tail_weighted_mse, calculate_tail_metrics

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _f1_score(precision, recall):
    """Compute F1 score from precision and recall."""
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

class HyperparameterTuner:
    def __init__(self, model_type, X_train, y_train, X_val, y_val, full_X=None, full_y=None):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # For walk-forward CV tuning of deep models
        # If full data is provided, use it; otherwise reconstruct from train+val
        if full_X is not None and full_y is not None:
            self.full_X = full_X
            self.full_y = full_y
        else:
            # Concatenate train and val, sort by index to ensure temporal order
            self.full_X = pd.concat([X_train, X_val]).sort_index()
            self.full_y = pd.concat([y_train, y_val]).sort_index()
    
    def _get_tuning_splits(self):
        """
        Generate (X_train, y_train, X_val, y_val) splits for hyperparameter tuning
        using a walk-forward scheme over [config.TUNE_START_DATE, config.TUNE_END_DATE].
        
        Returns:
            List of tuples: [(X_train, y_train, X_val, y_val), ...]
        """
        tune_start = pd.to_datetime(getattr(config, 'TUNE_START_DATE', '2012-01-01'))
        tune_end = pd.to_datetime(getattr(config, 'TUNE_END_DATE', '2022-12-31'))
        train_years = getattr(config, 'TUNE_TRAIN_YEARS', 5)
        val_months = getattr(config, 'TUNE_VAL_MONTHS', 6)
        step_months = getattr(config, 'TUNE_STEP_MONTHS', 6)
        buffer_days = getattr(config, 'TUNE_BUFFER_DAYS', 21)
        max_folds = getattr(config, 'TUNE_MAX_FOLDS', 10)
        
        # Restrict data to tuning window
        mask = (self.full_X.index >= tune_start) & (self.full_X.index <= tune_end)
        X_tune = self.full_X.loc[mask]
        y_tune = self.full_y.loc[mask]
        
        if len(X_tune) == 0:
            print("Warning: No data in tuning window. Falling back to single split.")
            return [(self.X_train, self.y_train, self.X_val, self.y_val)]
        
        # Use WalkForwardSplitter to generate folds
        # The splitter yields (fold, train_idx, val_idx, test_idx)
        # For tuning, we'll use val_idx as our validation and ignore test_idx
        splitter = data_prep.WalkForwardSplitter(
            start_date=tune_start + pd.DateOffset(years=train_years) + pd.DateOffset(months=val_months),
            train_years=train_years,
            val_months=val_months,
            buffer_days=buffer_days,
            step_months=step_months
        )
        
        # Create a combined dataframe for the splitter
        df_tune = pd.concat([X_tune, y_tune], axis=1)
        
        folds = []
        for fold_num, train_idx, val_idx, test_idx in splitter.split(df_tune):
            # Filter to only include folds where val ends before tune_end
            if len(val_idx) == 0 or val_idx.max() > tune_end:
                continue
                
            X_train_fold = X_tune.loc[train_idx]
            y_train_fold = y_tune.loc[train_idx]
            X_val_fold = X_tune.loc[val_idx]
            y_val_fold = y_tune.loc[val_idx]
            
            if len(X_train_fold) > 100 and len(X_val_fold) > 20:  # Minimum data requirements
                folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
            
            if len(folds) >= max_folds:
                break
        
        if len(folds) == 0:
            print("Warning: No valid tuning folds generated. Falling back to single split.")
            return [(self.X_train, self.y_train, self.X_val, self.y_val)]
        
        print(f"Generated {len(folds)} walk-forward folds for tuning")
        return folds
    
    def get_tuning_fold_count(self):
        """
        Get the number of folds that would be used for tuning.
        Useful for logging purposes.
        
        Returns:
            int: Number of folds, or None if not applicable
        """
        # Only deep models use walk-forward CV
        if self.model_type not in ['LSTM', 'CNN', 'Transformer']:
            return 1  # Static single split
        
        # Check if full data is available (indicates walk-forward CV)
        if not hasattr(self, 'full_X') or self.full_X is None:
            return 1  # Fallback to single split
        
        try:
            folds = self._get_tuning_splits()
            return len(folds)
        except Exception:
            return 1  # Fallback if something goes wrong
        
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
        elif self.model_type == 'Transformer':
            study.optimize(self._objective_transformer, n_trials=n_trials)
        else:
            print(f"Optuna not implemented for {self.model_type}. Skipping.")
            return {}
            
        print(f"Best params: {study.best_params}")
        print(f"Best objective (MSE - 0.5*F1): {study.best_value:.6f}")
        return study.best_params

    def _objective_lstm(self, trial):
        """Optuna objective for LSTM with walk-forward CV to avoid regime overfitting."""
        # Define Search Space
        time_steps = trial.suggest_categorical(
            'time_steps',
            getattr(config, 'LSTM_LOOKBACK_CANDIDATES', [5, 10, 20, 30, 45, 60])
        )
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Tail-weighted loss parameters
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail_alpha = getattr(config, 'BIG_MOVE_ALPHA', 4.0)
        epochs = getattr(config, 'TUNE_EPOCHS', 15)
        
        # Get walk-forward tuning folds
        folds = self._get_tuning_splits()
        
        # Collect metrics across all folds
        mse_list = []
        big_f1_up_list = []
        big_f1_down_list = []
        
        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            # Scale features: fit on train, apply to val (per fold)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Create sequences
            X_train_seq, y_train_seq = lstm_dataset.create_sequences(
                X_train_scaled, y_train_fold.values, time_steps
            )
            X_val_seq, y_val_seq = lstm_dataset.create_sequences(
                X_val_scaled, y_val_fold.values, time_steps
            )
            
            # Skip if insufficient data for this fold
            if len(X_train_seq) < batch_size or len(X_val_seq) == 0:
                continue
            
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
            
            # Initialize fresh model for each fold (no leakage across folds)
            input_dim = X_train_seq.shape[2]
            model = models.LSTMModel(input_dim, hidden_dim, num_layers, dropout=dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train Loop
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = tail_weighted_mse(outputs, y_batch, threshold=big_move_thresh, alpha=tail_alpha)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on this fold's validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
                y_pred = model(X_val_tensor).numpy().flatten()
            
            # Adjust actuals to match sequence output
            y_val_actual = y_val_fold.iloc[time_steps-1:]
            
            if len(y_val_actual) != len(y_pred):
                # Ensure alignment
                min_len = min(len(y_val_actual), len(y_pred))
                y_val_actual = y_val_actual.iloc[:min_len]
                y_pred = y_pred[:min_len]
            
            # Compute metrics for this fold
            mse = mean_squared_error(y_val_actual, y_pred)
            tail = calculate_tail_metrics(y_val_actual.values, y_pred, threshold=big_move_thresh)
            big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
            big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
            
            mse_list.append(mse)
            big_f1_up_list.append(big_f1_up)
            big_f1_down_list.append(big_f1_down)
        
        # Aggregate metrics across folds
        if len(mse_list) == 0:
            return float('inf')  # No valid folds
        
        mean_mse = np.mean(mse_list)
        mean_big_f1_up = np.mean(big_f1_up_list)
        mean_big_f1_down = np.mean(big_f1_down_list)
        
        # Combined objective: penalize MSE, reward big-move F1
        # Optuna minimizes, so subtract F1 terms
        objective = mean_mse - 0.5 * (mean_big_f1_up + mean_big_f1_down)
        return objective

    def _objective_xgboost(self, trial):
        from xgboost import XGBRegressor
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
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
        """Optuna objective for CNN with walk-forward CV to avoid regime overfitting."""
        # Define Search Space
        time_steps = trial.suggest_categorical(
            'time_steps',
            getattr(config, 'CNN_LOOKBACK_CANDIDATES', [5, 10, 20, 30, 45, 60])
        )
        filters = trial.suggest_categorical('filters', [32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        num_layers = trial.suggest_int('layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Tail-weighted loss parameters
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail_alpha = getattr(config, 'BIG_MOVE_ALPHA', 4.0)
        epochs = getattr(config, 'TUNE_EPOCHS', 15)
        
        # Get walk-forward tuning folds
        folds = self._get_tuning_splits()
        
        # Collect metrics across all folds
        mse_list = []
        big_f1_up_list = []
        big_f1_down_list = []
        
        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            # Scale features: fit on train, apply to val (per fold)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Create sequences
            X_train_seq, y_train_seq = lstm_dataset.create_sequences(
                X_train_scaled, y_train_fold.values, time_steps
            )
            X_val_seq, y_val_seq = lstm_dataset.create_sequences(
                X_val_scaled, y_val_fold.values, time_steps
            )
            
            # Skip if insufficient data for this fold
            if len(X_train_seq) < batch_size or len(X_val_seq) == 0:
                continue
            
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
            
            # Initialize fresh model for each fold (no leakage across folds)
            input_dim = X_train_seq.shape[2]
            model = models.CNN1DModel(input_dim, filters, kernel_size, num_layers, dropout=dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train Loop
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = tail_weighted_mse(outputs, y_batch, threshold=big_move_thresh, alpha=tail_alpha)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on this fold's validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
                y_pred = model(X_val_tensor).numpy().flatten()
            
            # Adjust actuals to match sequence output
            y_val_actual = y_val_fold.iloc[time_steps-1:]
            
            if len(y_val_actual) != len(y_pred):
                # Ensure alignment
                min_len = min(len(y_val_actual), len(y_pred))
                y_val_actual = y_val_actual.iloc[:min_len]
                y_pred = y_pred[:min_len]
            
            # Compute metrics for this fold
            mse = mean_squared_error(y_val_actual, y_pred)
            tail = calculate_tail_metrics(y_val_actual.values, y_pred, threshold=big_move_thresh)
            big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
            big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
            
            mse_list.append(mse)
            big_f1_up_list.append(big_f1_up)
            big_f1_down_list.append(big_f1_down)
        
        # Aggregate metrics across folds
        if len(mse_list) == 0:
            return float('inf')  # No valid folds
        
        mean_mse = np.mean(mse_list)
        mean_big_f1_up = np.mean(big_f1_up_list)
        mean_big_f1_down = np.mean(big_f1_down_list)
        
        # Combined objective: penalize MSE, reward big-move F1
        # Optuna minimizes, so subtract F1 terms
        objective = mean_mse - 0.5 * (mean_big_f1_up + mean_big_f1_down)
        return objective

    def _objective_transformer(self, trial):
        """Optuna objective for Transformer with walk-forward CV to avoid regime overfitting."""
        from .transformer import TransformerModel
        
        # Define Search Space
        time_steps = trial.suggest_categorical(
            'time_steps',
            getattr(config, 'TRANSFORMER_LOOKBACK_CANDIDATES', [10, 20, 30, 45, 60])
        )
        model_dim = trial.suggest_categorical('model_dim', [32, 64, 96])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Tail-weighted loss parameters
        big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
        tail_alpha = getattr(config, 'BIG_MOVE_ALPHA', 4.0)
        epochs = getattr(config, 'TUNE_EPOCHS', 15)
        
        # Get walk-forward tuning folds
        folds = self._get_tuning_splits()
        
        # Collect metrics across all folds
        mse_list = []
        big_f1_up_list = []
        big_f1_down_list = []
        
        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            # Scale features: fit on train, apply to val (per fold)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Create sequences
            X_train_seq, y_train_seq = lstm_dataset.create_sequences(
                X_train_scaled, y_train_fold.values, time_steps
            )
            X_val_seq, y_val_seq = lstm_dataset.create_sequences(
                X_val_scaled, y_val_fold.values, time_steps
            )
            
            # Skip if insufficient data for this fold
            if len(X_train_seq) < batch_size or len(X_val_seq) == 0:
                continue
            
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
            
            # Initialize fresh model for each fold (no leakage across folds)
            input_dim = X_train_seq.shape[2]
            model = TransformerModel(
                input_dim=input_dim,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Train Loop
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = tail_weighted_mse(outputs, y_batch, threshold=big_move_thresh, alpha=tail_alpha)
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # Evaluate on this fold's validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
                y_pred = model(X_val_tensor).numpy().flatten()
            
            # Adjust actuals to match sequence output
            y_val_actual = y_val_fold.iloc[time_steps-1:]
            
            if len(y_val_actual) != len(y_pred):
                # Ensure alignment
                min_len = min(len(y_val_actual), len(y_pred))
                y_val_actual = y_val_actual.iloc[:min_len]
                y_pred = y_pred[:min_len]
            
            # Compute metrics for this fold
            mse = mean_squared_error(y_val_actual, y_pred)
            tail = calculate_tail_metrics(y_val_actual.values, y_pred, threshold=big_move_thresh)
            big_f1_up = _f1_score(tail["precision_up_strict"], tail["recall_up_strict"])
            big_f1_down = _f1_score(tail["precision_down_strict"], tail["recall_down_strict"])
            
            mse_list.append(mse)
            big_f1_up_list.append(big_f1_up)
            big_f1_down_list.append(big_f1_down)
        
        # Aggregate metrics across folds
        if len(mse_list) == 0:
            return float('inf')  # No valid folds
        
        mean_mse = np.mean(mse_list)
        mean_big_f1_up = np.mean(big_f1_up_list)
        mean_big_f1_down = np.mean(big_f1_down_list)
        
        # Combined objective: penalize MSE, reward big-move F1
        # Optuna minimizes, so subtract F1 terms
        objective = mean_mse - 0.5 * (mean_big_f1_up + mean_big_f1_down)
        return objective
