import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils, metrics, lstm_dataset, tuning
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def evaluate(y_true, y_pred, set_name="Val"):
    """
    Calculates and prints evaluation metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Directional Accuracy
    # (Sign of Actual == Sign of Predicted)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    
    print(f"\n--- {set_name} Metrics ---")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    
    # Advanced Metrics
    ic = metrics.calculate_ic(y_true, y_pred)
    
    # Use config threshold for tail metrics
    big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
    pred_clip = getattr(config, 'PRED_CLIP', None)
    
    
    # Extract dates for monthly execution support
    dates = y_true.index if hasattr(y_true, 'index') else None
    
    # Strategy metrics (with optional prediction clipping)
    strat_metrics = metrics.calculate_strategy_metrics(y_true, y_pred, pred_clip=pred_clip, dates=dates)
    tail_metrics = metrics.calculate_tail_metrics(y_true, y_pred, threshold=big_move_thresh)
    
    # Big-move-only strategy metrics (with optional prediction clipping)
    bigmove_strat = metrics.calculate_bigmove_strategy_metrics(y_true, y_pred, threshold=big_move_thresh, pred_clip=pred_clip, dates=dates)
    
    print(f"IC: {ic:.4f}")
    print(f"\n  [Always-In Strategy]")
    print(f"  Sharpe (Ann.): {strat_metrics['sharpe']:.2f}")
    print(f"  Total Return: {strat_metrics['total_return']:.4f}")
    print(f"  Max Drawdown: {strat_metrics['max_drawdown']:.4f}")
    
    print(f"\n  [Big-Move-Only Strategy] (threshold={big_move_thresh:.1%})")
    print(f"  Sharpe (Ann.): {bigmove_strat['sharpe']:.2f}")
    print(f"  Total Return: {bigmove_strat['total_return']:.4f}")
    print(f"  Ann. Return: {bigmove_strat['ann_return']:.4f}")
    print(f"  Max Drawdown: {bigmove_strat['max_drawdown']:.4f}")
    print(f"  Trade Count: {bigmove_strat['trade_count']} ({bigmove_strat['holding_frequency']:.1%} of periods)")
    print(f"  Avg Return/Trade: {bigmove_strat['avg_return_per_trade']:.4f}")
    
    print(f"\n  [Big Move Detection]")
    print(f"  Precision (Up): {tail_metrics['precision_up_strict']:.2f}")
    print(f"  Recall (Up): {tail_metrics['recall_up_strict']:.2f}")
    print(f"  Precision (Down): {tail_metrics['precision_down_strict']:.2f}")
    print(f"  Recall (Down): {tail_metrics['recall_down_strict']:.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'dir_acc': dir_acc,
        'ic': ic,
        'strat_metrics': strat_metrics,
        'tail_metrics': tail_metrics,
        'bigmove_strat': bigmove_strat
    }


def resolve_model_params(model_type, best_params):
    """Merge config defaults with tuned parameters without mutating globals."""
    params = {}
    if model_type == 'RandomForest':
        params = {
            'n_estimators': config.RF_N_ESTIMATORS,
            'max_depth': config.RF_MAX_DEPTH,
            'min_samples_split': config.RF_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': config.RF_MIN_SAMPLES_LEAF,
            'random_state': config.RF_RANDOM_STATE,
            'n_jobs': -1
        }
    elif model_type == 'XGBoost':
        params = {
            'n_estimators': config.XGB_N_ESTIMATORS,
            'learning_rate': config.XGB_LEARNING_RATE,
            'max_depth': config.XGB_MAX_DEPTH,
            'min_child_weight': getattr(config, 'XGB_MIN_CHILD_WEIGHT', 1),
            'subsample': getattr(config, 'XGB_SUBSAMPLE', 1.0),
            'colsample_bytree': getattr(config, 'XGB_COLSAMPLE_BYTREE', 1.0),
            'gamma': getattr(config, 'XGB_GAMMA', 0),
            'reg_alpha': getattr(config, 'XGB_REG_ALPHA', 0),
            'reg_lambda': getattr(config, 'XGB_REG_LAMBDA', 1),
            'max_delta_step': getattr(config, 'XGB_MAX_DELTA_STEP', 0),
            'n_jobs': -1,
            'random_state': 42
        }
    elif model_type == 'MLP':
        params = {
            'hidden_layer_sizes': config.MLP_HIDDEN_LAYERS,
            'learning_rate_init': config.MLP_LEARNING_RATE_INIT,
            'alpha': config.MLP_ALPHA,
            'max_iter': config.MLP_MAX_ITER,
            'random_state': 42,
            'early_stopping': True
        }

        if 'n_layers' in best_params:
            n_layers = best_params.get('n_layers', len(config.MLP_HIDDEN_LAYERS))
            layers = []
            for i in range(n_layers):
                layers.append(best_params.get(f'n_units_l{i}', config.MLP_HIDDEN_LAYERS[i % len(config.MLP_HIDDEN_LAYERS)]))
            params['hidden_layer_sizes'] = tuple(layers)
        if 'learning_rate_init' in best_params:
            params['learning_rate_init'] = best_params['learning_rate_init']
        if 'alpha' in best_params:
            params['alpha'] = best_params['alpha']

    # Apply overrides where keys overlap
    for k, v in best_params.items():
        if k in params:
            params[k] = v

    return params


def resolve_deep_settings(model_type, best_params):
    """Prepare deep learning hyperparameters from config plus tuning results."""
    if model_type == 'LSTM':
        return {
            'hidden_dim': best_params.get('hidden_dim', config.LSTM_HIDDEN_DIM),
            'num_layers': best_params.get('num_layers', config.LSTM_LAYERS),
            'dropout': best_params.get('dropout', 0.2),
            'lr': best_params.get('lr', config.LSTM_LEARNING_RATE),
            'batch_size': best_params.get('batch_size', config.LSTM_BATCH_SIZE),
            'epochs': config.LSTM_EPOCHS
        }
    elif model_type == 'CNN':
        return {
            'filters': best_params.get('filters', config.CNN_FILTERS),
            'kernel_size': best_params.get('kernel_size', config.CNN_KERNEL_SIZE),
            'layers': best_params.get('layers', config.CNN_LAYERS),
            'dropout': best_params.get('dropout', config.CNN_DROPOUT),
            'lr': best_params.get('lr', config.CNN_LEARNING_RATE),
            'batch_size': best_params.get('batch_size', config.CNN_BATCH_SIZE),
            'epochs': config.CNN_EPOCHS
        }
    elif model_type == 'Transformer':
        return {
            'model_dim': best_params.get('model_dim', config.TRANSFORMER_MODEL_DIM),
            'num_heads': best_params.get('num_heads', config.TRANSFORMER_HEADS),
            'num_layers': best_params.get('num_layers', config.TRANSFORMER_LAYERS),
            'dim_feedforward': best_params.get('dim_feedforward', config.TRANSFORMER_FEEDFORWARD_DIM),
            'dropout': best_params.get('dropout', config.TRANSFORMER_DROPOUT),
            'lr': best_params.get('lr', config.TRANSFORMER_LR),
            'weight_decay': best_params.get('weight_decay', config.TRANSFORMER_WEIGHT_DECAY),
            'batch_size': best_params.get('batch_size', config.TRANSFORMER_BATCH_SIZE),
            'epochs': config.TRANSFORMER_EPOCHS
        }
    return {}

def main():
    # Initialize Logger with loss mode tag
    loss_tag = config.LOSS_MODE.upper()
    logger = utils.ExperimentLogger(model_name=config.MODEL_TYPE, process_tag="Static", loss_tag=loss_tag)

    # 1. Load Data using dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    # 2. Split Data with frequency-aware splitter
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        embargo_rows=embargo_rows,
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    # Features and Target (exclude BigMove labels to prevent target leakage)
    target_col = config.TARGET_COL
    exclude_cols = [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
    X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col]
    X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, target_col]
    
    print(f"\nTraining on {len(X_train)} samples, Validating on {len(X_val)}, Testing on {len(X_test)}")
    
    # 3. Train Model
    print(f"Training {config.MODEL_TYPE}...")
    
    # --- Optuna Tuning ---
    best_params = {}
    if config.USE_OPTUNA:
        # For deep models, pass full data to enable walk-forward CV during tuning
        # This avoids regime overfitting by evaluating hyperparameters across multiple folds
        if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
            full_X = df[feature_cols]
            full_y = df[target_col]
            tuner = tuning.HyperparameterTuner(
                config.MODEL_TYPE, X_train, y_train, X_val, y_val,
                full_X=full_X, full_y=full_y
            )
        else:
            tuner = tuning.HyperparameterTuner(config.MODEL_TYPE, X_train, y_train, X_val, y_val)
        best_params = tuner.optimize(n_trials=config.OPTUNA_TRIALS)
        
        # Record tuning metadata for logging
        if hasattr(logger, "set_tuning_info"):
            if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
                # Deep models use walk-forward CV
                n_folds = tuner.get_tuning_fold_count()
                logger.set_tuning_info(
                    method="WalkForward CV (multiple folds across tuning window)",
                    n_folds=n_folds,
                    tune_start_date=getattr(config, 'TUNE_START_DATE', None),
                    tune_end_date=getattr(config, 'TUNE_END_DATE', None),
                    n_trials=config.OPTUNA_TRIALS,
                )
            else:
                # Basic models use static single split
                logger.set_tuning_info(
                    method="Static (single train/val split)",
                    n_folds=1,
                    tune_start_date=str(X_train.index.min().date()) if hasattr(X_train.index, "min") else None,
                    tune_end_date=str(X_val.index.max().date()) if hasattr(X_val.index, "max") else None,
                    n_trials=config.OPTUNA_TRIALS,
                )
    else:
        if hasattr(logger, "set_tuning_info"):
            logger.set_tuning_info(
                method="None (no hyperparameter tuning)",
                n_trials=0,
            )
            
    # Track target scaling info for deep models (will be None for non-deep)
    target_scaling_info = None
    
    if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
        # --- Deep Learning Training Logic ---
        print(f"Preparing data for {config.MODEL_TYPE}...")

        # 1. Scale target for stability (fit on train only)
        scaling_mode = getattr(config, 'TARGET_SCALING_MODE', 'standardize')
        y_std = y_train.std()
        if y_std == 0 or y_std < 1e-8:
            y_std = 1.0
        
        if scaling_mode == "vol_scale":
            # Vol-scale: divide by std only (keeps 0 at 0)
            y_mean = 0.0
            y_train_scaled = y_train / y_std
            y_val_scaled = y_val / y_std
            y_test_scaled = y_test / y_std
        else:
            # Standardize: (y - mean) / std
            y_mean = y_train.mean()
            y_train_scaled = (y_train - y_mean) / y_std
            y_val_scaled = (y_val - y_mean) / y_std
            y_test_scaled = (y_test - y_mean) / y_std
        
        target_scaling_info = {'y_mean': float(y_mean), 'y_std': float(y_std), 'mode': scaling_mode}
        print(f"Target scaling ({scaling_mode}): mean={y_mean:.6f}, std={y_std:.6f}")
        
        # 2. Scale Features (Fit on Train, Transform on Val/Test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 3. Determine time_steps (from config default or tuned value)
        if config.MODEL_TYPE == 'LSTM':
            time_steps = getattr(config, 'LSTM_TIME_STEPS', 10)
        elif config.MODEL_TYPE == 'CNN':
            time_steps = getattr(config, 'CNN_TIME_STEPS', 10)
        elif config.MODEL_TYPE == 'Transformer':
            time_steps = getattr(config, 'TRANSFORMER_TIME_STEPS', 20)
        
        # Override with tuned value if available
        if best_params and 'time_steps' in best_params:
            time_steps = best_params['time_steps']
        
        print(f"Using time_steps={time_steps} for sequence creation")

        # 4. Reshape to 3D Sequences (using scaled targets)
        X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, y_train_scaled.values, time_steps)
        X_val_seq, y_val_seq = lstm_dataset.create_sequences(X_val_scaled, y_val_scaled.values, time_steps)
        X_test_seq, y_test_seq = lstm_dataset.create_sequences(X_test_scaled, y_test_scaled.values, time_steps)

        # 5. Create DataLoaders
        deep_settings = resolve_deep_settings(config.MODEL_TYPE, best_params)
        batch_size = deep_settings.get('batch_size', config.LSTM_BATCH_SIZE)
        train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)

        # 6. Initialize Model
        input_dim = X_train_seq.shape[2]
        if config.MODEL_TYPE == 'LSTM':
            model = models.LSTMModel(
                input_dim,
                deep_settings['hidden_dim'],
                deep_settings['num_layers'],
                dropout=deep_settings['dropout']
            )
            lr = deep_settings['lr']
            weight_decay = 0
            epochs = deep_settings['epochs']
        elif config.MODEL_TYPE == 'CNN':
            model = models.CNN1DModel(
                input_dim,
                deep_settings['filters'],
                deep_settings['kernel_size'],
                deep_settings['layers'],
                dropout=deep_settings['dropout']
            )
            lr = deep_settings['lr']
            weight_decay = 0
            epochs = deep_settings['epochs']
        elif config.MODEL_TYPE == 'Transformer':
            from ML.transformer import TransformerModel
            model = TransformerModel(
                input_dim=input_dim,
                model_dim=deep_settings['model_dim'],
                num_heads=deep_settings['num_heads'],
                num_layers=deep_settings['num_layers'],
                dim_feedforward=deep_settings['dim_feedforward'],
                dropout=deep_settings['dropout']
            )
            lr = deep_settings['lr']
            weight_decay = deep_settings['weight_decay']
            epochs = deep_settings['epochs']

        # Loss function configuration (scale threshold for scaled targets)
        loss_mode = config.LOSS_MODE
        huber_delta = getattr(config, 'HUBER_DELTA', 1.0)
        tail_alpha = getattr(config, 'TAIL_ALPHA', 4.0)
        tail_threshold_raw = getattr(config, 'TAIL_THRESHOLD', 0.03)
        tail_threshold_scaled = tail_threshold_raw / y_std  # Scale threshold for scaled targets
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate warmup scheduler for Transformer stability
        scheduler = None
        if config.MODEL_TYPE == 'Transformer':
            warmup_epochs = 5
            warmup_steps = len(train_loader) * warmup_epochs
            step_count = [0]  # Use list to allow mutation in closure
            def lr_lambda(step):
                step_count[0] += 1
                if step_count[0] < warmup_steps:
                    return float(step_count[0]) / float(max(1, warmup_steps))
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 6. Train Loop
        print(f"Training {config.MODEL_TYPE} for {epochs} epochs (loss_mode={loss_mode})...")
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = utils.compute_loss(
                    y_pred=outputs,
                    y_true=y_batch,
                    loss_mode=loss_mode,
                    huber_delta=huber_delta,
                    tail_alpha=tail_alpha,
                    tail_threshold=tail_threshold_scaled
                )
                loss.backward()
                # Gradient clipping for stability (especially important for Transformer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss += loss.item()

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")

        # 7. Predict and Unscale
        model.eval()
        with torch.no_grad():
            # Convert Train/Val/Test to tensor
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            y_train_pred_scaled = model(X_train_tensor).numpy().flatten()

            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_val_pred_scaled = model(X_val_tensor).numpy().flatten()

            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
            y_test_pred_scaled = model(X_test_tensor).numpy().flatten()
        
        # Unscale predictions back to original scale
        # For vol_scale: y_mean is 0.0, so this simplifies to y_pred * y_std
        # For standardize: y_pred * y_std + y_mean
        y_train_pred = y_train_pred_scaled * y_std + y_mean
        y_val_pred = y_val_pred_scaled * y_std + y_mean
        y_test_pred = y_test_pred_scaled * y_std + y_mean

        # Adjust Actuals to match sequence length (first time_steps-1 are lost)
        # Use raw (unscaled) targets for evaluation
        y_train_actual = y_train.iloc[time_steps-1:]
        y_val_actual = y_val.iloc[time_steps-1:]
        y_test_actual = y_test.iloc[time_steps-1:]

        # Override for evaluation
        y_train = y_train_actual
        y_val = y_val_actual
        y_test = y_test_actual
        
    else:
        # --- Standard ML Training ---
        model_params = resolve_model_params(config.MODEL_TYPE, best_params)
        model = models.ModelFactory.get_model(config.MODEL_TYPE, overrides=model_params)
        
        # Compute sample weights to emphasize big moves
        # Use TAIL_ALPHA if available, otherwise fall back to BIG_MOVE_ALPHA for backward compatibility
        tail_alpha = getattr(config, 'TAIL_ALPHA', getattr(config, 'BIG_MOVE_ALPHA', 4.0))
        
        if tail_alpha == 0.0:
            # No weighting needed - use uniform weights to avoid unnecessary computation
            sample_weight = np.ones(len(y_train))
            print("Sample weighting: Disabled (TAIL_ALPHA = 0.0)")
        else:
            big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
            is_big = (y_train.abs() > big_move_thresh).astype(float)
            sample_weight = 1.0 + tail_alpha * is_big
            n_big = int(is_big.sum())
            print(f"Sample weighting: {n_big} big moves ({100*n_big/len(y_train):.1f}%) get {1+tail_alpha}x weight")
        
        # Try to fit with sample_weight; fall back if not supported
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            print(f"Note: {config.MODEL_TYPE} does not support sample_weight, fitting without weights.")
            model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
    
    # 4. Evaluate
    # Validation
    # Validation
    metrics_val = evaluate(y_val, y_val_pred, "Validation")
    
    # Test (Out of Sample)
    metrics_test = evaluate(y_test, y_test_pred, "Test (OOS)")
    
    # Dummy train metrics for logging (optional, or calculate real ones)
    metrics_train = {'rmse': 0, 'mae': 0, 'dir_acc': 0} 
    
    # 5. Plot
    # Function to plot time series
    def plot_ts(y_true, y_pred, title, filename):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true, label='Actual', alpha=0.7)
        plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        logger.save_plot(fig, filename)
        plt.close(fig)

    # Train Plot
    plot_ts(y_train, y_train_pred, f"Train Set: Actual vs Predicted ({config.MODEL_TYPE})", "plot_train.png")
    
    # Val Plot
    plot_ts(y_val, y_val_pred, f"Validation Set: Actual vs Predicted ({config.MODEL_TYPE})", "plot_val.png")
    
    # Test Plot
    plot_ts(y_test, y_test_pred, f"Test Set: Actual vs Predicted ({config.MODEL_TYPE})", "plot_test.png")
    
    # Scatter Plot (Test)
    logger.plot_scatter(y_test, y_test_pred, title=f"Test Set Scatter: {config.MODEL_TYPE}", filename="scatter_test.png")
    
    # Log Results (pass test arrays for diagnostics)
    # Convert to numpy for diagnostics
    y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_test_pred_array = np.array(y_test_pred)
    
    # Save configuration JSON for reproducibility
    logger.save_config_json(config.MODEL_TYPE, best_params)
    
    logger.log_summary(
        metrics_train, metrics_val, metrics_test, 
        config.MODEL_TYPE, feature_cols,
        y_true=y_test_array, 
        y_pred=y_test_pred_array,
        target_scaling_info=target_scaling_info
    )
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        print("\nTop 10 Features:")
        print(importances.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type to train')
    parser.add_argument('--optimize', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    args = parser.parse_args()
    
    if args.model_type:
        config.MODEL_TYPE = args.model_type
        
    if args.optimize:
        config.USE_OPTUNA = True
        print(f"Hyperparameter Tuning Enabled (Trials: {args.trials})")
        
    if args.trials:
        config.OPTUNA_TRIALS = args.trials
        
    main()
