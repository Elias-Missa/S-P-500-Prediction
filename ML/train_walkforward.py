import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils, metrics, lstm_dataset, tuning
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def main():
    # Initialize Logger with loss mode tag
    loss_tag = config.LOSS_MODE.upper()
    logger = utils.ExperimentLogger(model_name=config.MODEL_TYPE, process_tag="WalkForward", loss_tag=loss_tag)
    
    print("--- Walk-Forward Validation ---")
    
    # 1. Load Data using dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    print(f"Dataset: frequency={frequency}, embargo_rows={embargo_rows}")
    
    # 2. Setup Splitter with frequency-aware configuration
    # For monthly data, use larger step (3 months) to ensure enough test samples
    # For daily data, step_months=1 is fine
    step_months = 3 if frequency == "monthly" else 1
    
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,  # Use walk-forward specific val_months
        embargo_rows=embargo_rows,
        step_months=step_months,
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    print(f"Walk-forward step: {step_months} months")
    
    # Load tuned hyperparameters if configured
    if config.WF_USE_TUNED_PARAMS and config.WF_BEST_PARAMS_PATH:
        if os.path.exists(config.WF_BEST_PARAMS_PATH):
            with open(config.WF_BEST_PARAMS_PATH, 'r') as f:
                tuned = json.load(f)
            print(f"Loaded tuned params from {config.WF_BEST_PARAMS_PATH}")
            
            # Override Transformer params - support Optuna keys with backward compatibility
            # Prefer Optuna keys (num_heads, num_layers, dim_feedforward) over old keys (heads, layers, ff_dim)
            if 'model_dim' in tuned:
                config.TRANSFORMER_MODEL_DIM = tuned['model_dim']
            
            # num_heads (Optuna) or heads (backward compat)
            if 'num_heads' in tuned:
                config.TRANSFORMER_HEADS = tuned['num_heads']
            elif 'heads' in tuned:
                config.TRANSFORMER_HEADS = tuned['heads']
            
            # num_layers (Optuna) or layers (backward compat)
            if 'num_layers' in tuned:
                config.TRANSFORMER_LAYERS = tuned['num_layers']
            elif 'layers' in tuned:
                config.TRANSFORMER_LAYERS = tuned['layers']
            
            # dim_feedforward (Optuna) or ff_dim (backward compat)
            if 'dim_feedforward' in tuned:
                config.TRANSFORMER_FEEDFORWARD_DIM = tuned['dim_feedforward']
            elif 'ff_dim' in tuned:
                config.TRANSFORMER_FEEDFORWARD_DIM = tuned['ff_dim']
            
            if 'dropout' in tuned:
                config.TRANSFORMER_DROPOUT = tuned['dropout']
            if 'lr' in tuned:
                config.TRANSFORMER_LR = tuned['lr']
            if 'weight_decay' in tuned:
                config.TRANSFORMER_WEIGHT_DECAY = tuned['weight_decay']
            if 'batch_size' in tuned:
                config.TRANSFORMER_BATCH_SIZE = tuned['batch_size']
            if 'time_steps' in tuned:
                config.TRANSFORMER_TIME_STEPS = tuned['time_steps']
            
            # Print final overrides in a clean format
            print("\n--- Final Transformer Overrides ---")
            print(f"  model_dim:        {config.TRANSFORMER_MODEL_DIM}")
            print(f"  num_heads:        {config.TRANSFORMER_HEADS}")
            print(f"  num_layers:       {config.TRANSFORMER_LAYERS}")
            print(f"  dim_feedforward:  {config.TRANSFORMER_FEEDFORWARD_DIM}")
            print(f"  dropout:          {config.TRANSFORMER_DROPOUT}")
            print(f"  lr:               {config.TRANSFORMER_LR}")
            print(f"  weight_decay:    {config.TRANSFORMER_WEIGHT_DECAY}")
            print(f"  batch_size:       {config.TRANSFORMER_BATCH_SIZE}")
            print(f"  time_steps:       {config.TRANSFORMER_TIME_STEPS}")
            print("---\n")
        else:
            print(f"Warning: WF_BEST_PARAMS_PATH '{config.WF_BEST_PARAMS_PATH}' not found. Using defaults.")
    
    # Exclude BigMove labels to prevent target leakage
    target_col = config.TARGET_COL
    exclude_cols = [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    all_preds = []
    all_actuals = []
    fold_metrics_list = []  # Store per-fold metrics
    threshold_tuning_results = []  # Store per-fold threshold tuning results
    
    print(f"Model: {config.MODEL_TYPE}")
    if config.WF_TUNE_THRESHOLD:
        print(f"Threshold Tuning: ENABLED (criterion={config.WF_THRESHOLD_CRITERION}, grid_points={config.WF_THRESHOLD_N_GRID})")
    print(f"Rolling Train Window: {config.TRAIN_WINDOW_YEARS} years")
    
    # --- Optuna Hyperparameter Tuning (if enabled and not using pre-tuned params) ---
    best_params = {}
    if config.WF_USE_TUNED_PARAMS and config.WF_BEST_PARAMS_PATH and os.path.exists(config.WF_BEST_PARAMS_PATH):
        # Load pre-tuned params from file
        with open(config.WF_BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        print(f"Using pre-tuned parameters from {config.WF_BEST_PARAMS_PATH}")
        logger.set_tuning_info(
            method=f"Pre-tuned parameters loaded from file",
            n_folds=None,
            tune_start_date=None,
            tune_end_date=None,
            n_trials=None,
        )
    elif config.USE_OPTUNA:
        print("\n=== Starting Optuna Hyperparameter Tuning ===")
        # Use the first fold's data for tuning, or use tuning window if available
        # Get first fold for tuning data
        first_fold_data = list(splitter.split(df))
        if len(first_fold_data) > 0:
            first_fold, first_train_idx, first_val_idx, first_test_idx = first_fold_data[0]
            X_tune_train = df.loc[first_train_idx, feature_cols]
            y_tune_train = df.loc[first_train_idx, target_col]
            
            # For validation, use first fold's validation set if available, otherwise create a split
            if len(first_val_idx) > 0 and not config.WF_TRAIN_ON_TRAIN_PLUS_VAL:
                X_tune_val = df.loc[first_val_idx, feature_cols]
                y_tune_val = df.loc[first_val_idx, target_col]
            else:
                # Split training data for tuning validation
                split_point = int(len(X_tune_train) * 0.8)
                X_tune_val = X_tune_train.iloc[split_point:]
                y_tune_val = y_tune_train.iloc[split_point:]
                X_tune_train = X_tune_train.iloc[:split_point]
                y_tune_train = y_tune_train.iloc[:split_point]
            
            # For deep models, pass full data to enable walk-forward CV during tuning
            if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
                # Use data up to TEST_START_DATE for tuning (to avoid leakage)
                # Use row-based embargo: find test start position and go back EMBARGO_ROWS
                # For tuning, we approximate using business days since we don't have full df here
                tune_end_date = pd.to_datetime(config.TEST_START_DATE) - pd.offsets.BDay(config.EMBARGO_ROWS)
                mask = df.index <= tune_end_date
                full_X = df.loc[mask, feature_cols]
                full_y = df.loc[mask, target_col]
                tuner = tuning.HyperparameterTuner(
                    config.MODEL_TYPE, X_tune_train, y_tune_train, X_tune_val, y_tune_val,
                    full_X=full_X, full_y=full_y
                )
            else:
                tuner = tuning.HyperparameterTuner(
                    config.MODEL_TYPE, X_tune_train, y_tune_train, X_tune_val, y_tune_val
                )
            
            best_params = tuner.optimize(n_trials=config.OPTUNA_TRIALS)
            
            # Record tuning metadata for logging
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
                    tune_start_date=str(X_tune_train.index.min().date()) if hasattr(X_tune_train.index, "min") else None,
                    tune_end_date=str(X_tune_val.index.max().date()) if hasattr(X_tune_val.index, "max") else None,
                    n_trials=config.OPTUNA_TRIALS,
                )
            
            print(f"Optuna tuning complete. Best params: {best_params}\n")
        else:
            print("Warning: No folds available for tuning. Skipping Optuna optimization.")
    else:
        # Record that no tuning was performed
        logger.set_tuning_info(
            method="None (using default config parameters)",
            n_folds=None,
            tune_start_date=None,
            tune_end_date=None,
            n_trials=None,
        )
    
    # Materialize splits so we can report progress and counts
    splits = list(splitter.split(df))
    total_folds = len(splits)
    
    # Collect predictions and actuals for train, val, and test sets
    all_train_preds = []
    all_train_actuals = []
    all_val_preds = []
    all_val_actuals = []
    all_test_dates = []  # Collect dates for monthly execution mode
    
    for fold, train_idx, val_idx, test_idx in splits:
        print(f"\n=== Fold {fold+1}/{total_folds} ===")
        # Prepare Data
        X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
        X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, target_col]
        
        # Track whether we have a separate val set for metrics
        has_val_set = len(val_idx) > 0 and not config.WF_TRAIN_ON_TRAIN_PLUS_VAL
        X_val_orig, y_val_orig = None, None
        
        if len(val_idx) > 0:
            X_val_orig, y_val_orig = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col]
        
        # Merge train+val if configured and val is non-empty
        if config.WF_TRAIN_ON_TRAIN_PLUS_VAL and len(val_idx) > 0:
            print(f"  Merging train ({len(X_train)}) + val ({len(X_val_orig)}) for training")
            X_train = pd.concat([X_train, X_val_orig])
            y_train = pd.concat([y_train, y_val_orig])
            has_val_set = False  # No longer have separate val set
        
        # Initialize train predictions/actuals (will be set in both branches)
        y_train_pred = None
        y_train_actual = None
        
        # Train
        if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
            # --- Deep Learning Training Logic ---
            # Determine time_steps for deep models (use tuned params if available)
            time_steps = None
            if config.MODEL_TYPE == 'LSTM':
                time_steps = best_params.get('time_steps', getattr(config, 'LSTM_TIME_STEPS', 10))
            elif config.MODEL_TYPE == 'CNN':
                time_steps = best_params.get('time_steps', getattr(config, 'CNN_TIME_STEPS', 10))
            elif config.MODEL_TYPE == 'Transformer':
                time_steps = best_params.get('time_steps', getattr(config, 'TRANSFORMER_TIME_STEPS', 20))
            
            if time_steps is not None and fold == 0:
                print(f"Using time_steps={time_steps} for sequence creation")
            
            # 1. Scale target per fold for stability (train-only stats)
            scaling_mode = getattr(config, 'TARGET_SCALING_MODE', 'standardize')
            y_std = y_train.std()
            if y_std == 0 or y_std < 1e-8:
                y_std = 1.0
            
            if scaling_mode == "vol_scale":
                # Vol-scale: divide by std only (keeps 0 at 0)
                y_mean = 0.0
                y_train_scaled = y_train / y_std
                y_test_scaled = y_test / y_std
            else:
                # Standardize: (y - mean) / std
                y_mean = y_train.mean()
                y_train_scaled = (y_train - y_mean) / y_std
                y_test_scaled = (y_test - y_mean) / y_std
            
            print(f"  Fold {fold} target scaling ({scaling_mode}): y_mean={y_mean:.6f}, y_std={y_std:.6f}")
            
            # 2. Scale Features (Fit on Train, Transform on Test)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. Reshape to 3D Sequences (using scaled targets)
            X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, y_train_scaled.values, time_steps)
            X_test_seq, y_test_seq = lstm_dataset.create_sequences(X_test_scaled, y_test_scaled.values, time_steps)
            
            # Check if sequences are empty (e.g. if fold is too small)
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                print(f"Fold {fold}: Skipping due to insufficient data for sequence generation.")
                continue
            
            # 4. Create DataLoader (use tuned batch_size if available)
            if config.MODEL_TYPE == 'LSTM':
                batch_size = best_params.get('batch_size', config.LSTM_BATCH_SIZE)
            elif config.MODEL_TYPE == 'CNN':
                batch_size = best_params.get('batch_size', config.CNN_BATCH_SIZE)
            elif config.MODEL_TYPE == 'Transformer':
                batch_size = best_params.get('batch_size', config.TRANSFORMER_BATCH_SIZE)
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
            
            # 4b. Prepare Validation DataLoader for Early Stopping (if available)
            val_loader = None
            has_val = has_val_set and X_val_orig is not None and len(X_val_orig) >= time_steps
            if fold == 0 and config.MODEL_TYPE == 'Transformer':
                # Debug output for first fold to diagnose validation setup
                print(f"  Debug: has_val_set={has_val_set}, X_val_orig is not None={X_val_orig is not None}, "
                      f"len(X_val_orig)={len(X_val_orig) if X_val_orig is not None else 0}, "
                      f"time_steps={time_steps}, has_val={has_val}")
            if has_val:
                # Scale val features using train-fitted scaler
                X_val_scaled = scaler.transform(X_val_orig)
                
                # Scale val targets using train stats
                if scaling_mode == "vol_scale":
                    y_val_scaled = y_val_orig / y_std
                else:
                    y_val_scaled = (y_val_orig - y_mean) / y_std
                
                # Create sequences for val
                X_val_seq, y_val_seq = lstm_dataset.create_sequences(X_val_scaled, y_val_scaled.values, time_steps)
                
                if len(X_val_seq) > 0:
                    val_loader = lstm_dataset.prepare_dataloader(X_val_seq, y_val_seq, batch_size=batch_size)
                    print(f"  Validation set prepared: {len(X_val_seq)} sequences")
                else:
                    has_val = False
                    print(f"  Warning: Validation sequences empty, disabling early stopping for this fold")
            
            # 5. Initialize Model (apply tuned parameters)
            input_dim = X_train_seq.shape[2]
            weight_decay = 0
            if config.MODEL_TYPE == 'LSTM':
                # Apply tuned params to LSTM model
                hidden_dim = best_params.get('hidden_dim', config.LSTM_HIDDEN_DIM)
                num_layers = best_params.get('num_layers', config.LSTM_LAYERS)
                dropout = best_params.get('dropout', 0.2)
                from ML.models import LSTMModel
                model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1, dropout=dropout)
                lr = best_params.get('lr', config.LSTM_LEARNING_RATE)
                epochs = config.LSTM_EPOCHS
            elif config.MODEL_TYPE == 'CNN':
                # Apply tuned params to CNN model
                filters = best_params.get('filters', config.CNN_FILTERS)
                kernel_size = best_params.get('kernel_size', config.CNN_KERNEL_SIZE)
                num_layers = best_params.get('layers', config.CNN_LAYERS)
                dropout = best_params.get('dropout', config.CNN_DROPOUT)
                from ML.models import CNN1DModel
                model = CNN1DModel(input_dim, filters, kernel_size, num_layers, dropout=dropout, output_dim=1)
                lr = best_params.get('lr', config.CNN_LEARNING_RATE)
                epochs = config.CNN_EPOCHS
            elif config.MODEL_TYPE == 'Transformer':
                # Apply tuned params to Transformer model
                model_dim = best_params.get('model_dim', config.TRANSFORMER_MODEL_DIM)
                num_heads = best_params.get('num_heads', config.TRANSFORMER_HEADS)
                num_layers = best_params.get('num_layers', config.TRANSFORMER_LAYERS)
                dim_feedforward = best_params.get('dim_feedforward', config.TRANSFORMER_FEEDFORWARD_DIM)
                dropout = best_params.get('dropout', config.TRANSFORMER_DROPOUT)
                from ML.transformer.model import TransformerModel
                model = TransformerModel(
                    input_dim=input_dim,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                lr = best_params.get('lr', config.TRANSFORMER_LR)
                weight_decay = best_params.get('weight_decay', config.TRANSFORMER_WEIGHT_DECAY)
                epochs = config.TRANSFORMER_EPOCHS
            
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
            
            # 6. Training Loop with Early Stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None  # To save the best weights
            
            model.train()
            
            for epoch in range(epochs):
                # 1. Training Step
                epoch_loss = 0.0
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
                    # Apply gradient clipping from config
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(config, 'WF_GRAD_CLIP_NORM', 1.0))
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    epoch_loss += loss.item()
                
                # 2. Validation & Early Stopping
                if has_val and val_loader:
                    model.eval()
                    val_loss_sum = 0.0
                    with torch.no_grad():
                        for X_v, y_v in val_loader:
                            out_v = model(X_v)
                            loss_v = utils.compute_loss(
                                y_pred=out_v,
                                y_true=y_v,
                                loss_mode=loss_mode,
                                huber_delta=huber_delta,
                                tail_alpha=tail_alpha,
                                tail_threshold=tail_threshold_scaled
                            )
                            val_loss_sum += loss_v.item()
                    
                    avg_val_loss = val_loss_sum / max(1, len(val_loader))
                    avg_train_loss = epoch_loss / max(1, len(train_loader))
                    # Log progress occasionally (every 10 epochs or first one)
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
                    # Check Early Stopping
                    if getattr(config, 'WF_EARLY_STOPPING', False):
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            # Save best weights in memory
                            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        else:
                            patience_counter += 1
                            if patience_counter >= getattr(config, 'WF_PATIENCE', 15):
                                print(f"  [Early Stopping] No improvement for {getattr(config, 'WF_PATIENCE', 15)} epochs. Stopping at epoch {epoch+1}.")
                                break
                    
                    model.train()  # Switch back to train for next epoch
                else:
                    # No validation set - use simple logging
                    avg_loss = epoch_loss / max(1, len(train_loader))
                    # Log every 10 epochs (same as validation branch) or first/last epoch
                    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            # 3. Restore Best Weights (Critical!)
            if getattr(config, 'WF_EARLY_STOPPING', False) and best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"  Restored best model from validation (Loss: {best_val_loss:.6f})")
            
            # 7. Predict and Unscale
            model.eval()
            with torch.no_grad():
                # Predict on train set
                X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
                y_train_pred_scaled = model(X_train_tensor).numpy().flatten()
                y_train_pred = y_train_pred_scaled * y_std + y_mean
                y_train_actual = y_train.iloc[time_steps-1:].values
                
                # Predict on test set
                X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                y_pred_scaled = model(X_test_tensor).numpy().flatten()
            
            # Unscale predictions back to original scale
            # For vol_scale: y_mean is 0.0, so this simplifies to y_pred * y_std
            # For standardize: y_pred * y_std + y_mean
            y_pred = y_pred_scaled * y_std + y_mean
                
            # Adjust Actuals (slice off first time_steps-1) - use raw y_test for metrics
            y_test = y_test.iloc[time_steps-1:]
            
            # --- Compute Validation Metrics if val set is available ---
            y_val_pred, y_val_actual = None, None
            if has_val_set and X_val_orig is not None and len(X_val_orig) >= time_steps:
                # Scale val features using train-fitted scaler
                X_val_scaled = scaler.transform(X_val_orig)
                
                # Scale val targets using train stats
                if scaling_mode == "vol_scale":
                    y_val_scaled = y_val_orig / y_std
                else:
                    y_val_scaled = (y_val_orig - y_mean) / y_std
                
                # Create sequences for val
                X_val_seq, _ = lstm_dataset.create_sequences(X_val_scaled, y_val_scaled.values, time_steps)
                
                if len(X_val_seq) > 0:
                    # Predict on val
                    with torch.no_grad():
                        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
                        y_val_pred_scaled = model(X_val_tensor).numpy().flatten()
                    
                    # Unscale val predictions
                    y_val_pred = y_val_pred_scaled * y_std + y_mean
                    
                    # Slice val actuals to match sequence output
                    y_val_actual = y_val_orig.iloc[time_steps-1:].values
            
        else:
            # --- Standard ML Training ---
            # Apply tuned parameters for standard ML models
            # Helper function to resolve model params (copied from train.py for independence)
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
            
            model_params = resolve_model_params(config.MODEL_TYPE, best_params)
            model = models.ModelFactory.get_model(config.MODEL_TYPE, overrides=model_params)
            model.fit(X_train, y_train)
            
            # Predict on train, val, and test sets
            y_train_pred = model.predict(X_train)
            y_train_actual = y_train.values if hasattr(y_train, 'values') else y_train
            y_pred = model.predict(X_test)
            
            # --- Compute Validation Metrics for standard ML if val set is available ---
            y_val_pred, y_val_actual = None, None
            if has_val_set and X_val_orig is not None and len(X_val_orig) > 0:
                y_val_pred = model.predict(X_val_orig)
                y_val_actual = y_val_orig.values
        
        # Store predictions and actuals
        if y_train_pred is not None and y_train_actual is not None:
            all_train_preds.extend(y_train_pred)
            all_train_actuals.extend(y_train_actual)
        all_preds.extend(y_pred)
        all_actuals.extend(y_test)
        
        # Store test dates for monthly execution mode
        # Get dates from df index using test_idx
        try:
            # test_idx should be integer positions, use them to index df
            if hasattr(test_idx, '__iter__') and not isinstance(test_idx, str):
                # Convert to list if it's a range or other iterable
                import pandas as pd
                if isinstance(test_idx, (list, np.ndarray)) or (hasattr(pd, 'Index') and isinstance(test_idx, pd.Index)):
                    test_dates = df.index[test_idx]
                    all_test_dates.extend(test_dates.tolist())
                else:
                    # Try to convert to list
                    test_dates = df.index[list(test_idx)]
                    all_test_dates.extend(test_dates.tolist())
            elif hasattr(y_test, 'index'):
                all_test_dates.extend(y_test.index.tolist())
        except (IndexError, TypeError) as e:
            # Fallback: use y_test index if available
            if hasattr(y_test, 'index'):
                all_test_dates.extend(y_test.index.tolist())
            else:
                # If all else fails, skip date collection for this fold
                pass
        
        # Store validation predictions if available
        if y_val_pred is not None and y_val_actual is not None:
            all_val_preds.extend(y_val_pred)
            all_val_actuals.extend(y_val_actual)
        
        # Compute per-fold metrics
        y_test_arr = np.array(y_test) if hasattr(y_test, 'values') else y_test
        y_pred_arr = np.array(y_pred)
        
        if len(y_test_arr) > 0:
            fold_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
            fold_mae = mean_absolute_error(y_test_arr, y_pred_arr)
            fold_dir_acc = np.mean(np.sign(y_test_arr) == np.sign(y_pred_arr)) * 100
            fold_ic = metrics.calculate_ic(y_test_arr, y_pred_arr)
            
            big_move_thresh = getattr(config, 'TAIL_THRESHOLD', getattr(config, 'BIG_MOVE_THRESHOLD', 0.03))
            pred_clip = getattr(config, 'PRED_CLIP', None)
            execution_frequency = getattr(config, 'EXECUTION_FREQUENCY', 'daily')
            
            # Get fold dates for monthly execution mode
            fold_test_dates = None
            if execution_frequency == "monthly":
                try:
                    # Try to get dates from y_test index first (most reliable)
                    if hasattr(y_test, 'index') and len(y_test.index) == len(y_test_arr):
                        fold_test_dates = pd.DatetimeIndex(y_test.index)
                    # Otherwise try to get from df using test_idx
                    elif hasattr(test_idx, '__len__'):
                        # Convert test_idx to list if needed
                        idx_list = list(test_idx) if not isinstance(test_idx, (list, np.ndarray)) else test_idx
                        fold_test_dates = pd.DatetimeIndex(df.index[idx_list])
                except Exception as e:
                    # If date collection fails, monthly mode won't work for this fold
                    print(f"  Warning: Could not collect dates for fold {fold}: {e}")
                    fold_test_dates = None
            
            fold_tail = metrics.calculate_tail_metrics(y_test_arr, y_pred_arr, threshold=big_move_thresh)
            fold_strat = metrics.calculate_strategy_metrics(
                y_test_arr, y_pred_arr, pred_clip=pred_clip,
                dates=fold_test_dates, execution_frequency=execution_frequency
            )
            fold_bigmove_strat = metrics.calculate_bigmove_strategy_metrics(
                y_test_arr, y_pred_arr, threshold=big_move_thresh, pred_clip=pred_clip,
                dates=fold_test_dates, execution_frequency=execution_frequency
            )
            
            # --- Signal Concentration Analysis (Decile Spread + Coverage) ---
            fold_decile = metrics.calculate_decile_analysis(y_test_arr, y_pred_arr)
            fold_coverage = metrics.calculate_coverage_performance(
                y_test_arr, y_pred_arr, frequency=frequency,
                dates=fold_test_dates, execution_frequency=execution_frequency
            )
            
            fold_entry = {
                'fold_id': fold,
                'train_start': str(train_idx.min().date()),
                'train_end': str(train_idx.max().date()),
                'test_start': str(test_idx.min().date()),
                'test_end': str(test_idx.max().date()),
                'n_train': len(train_idx),
                'n_test': len(y_test_arr),
                'rmse': fold_rmse,
                'mae': fold_mae,
                'dir_acc': fold_dir_acc,
                'ic': fold_ic,
                # Decile spread metrics
                'decile_spread': fold_decile['spread'],
                'decile_spread_tstat': fold_decile['spread_tstat'],
                'decile_monotonicity': fold_decile['monotonicity'],
                'top_decile_mean': fold_decile['top_mean'],
                'bottom_decile_mean': fold_decile['bottom_mean'],
                # Coverage vs performance
                'best_thresh_sharpe': fold_coverage['best_sharpe'],
                'best_thresh': fold_coverage['best_sharpe_threshold'],
                'best_thresh_coverage': fold_coverage['best_sharpe_coverage'],
                # Existing metrics
                'big_up_precision': fold_tail['precision_up_strict'],
                'big_up_recall': fold_tail['recall_up_strict'],
                'big_down_precision': fold_tail['precision_down_strict'],
                'big_down_recall': fold_tail['recall_down_strict'],
                'strategy_sharpe': fold_strat['sharpe'],
                'big_move_sharpe': fold_bigmove_strat['sharpe']
            }

            # --- Regime-Specific Metrics (if RegimeGatedRidge or RegimeGatedHybrid) ---
            if config.MODEL_TYPE in ['RegimeGatedRidge', 'RegimeGatedHybrid'] and hasattr(model, 'regime_threshold'):
                regime_col = getattr(config, 'REGIME_COL', 'RV_Ratio')
                if regime_col in X_test.columns:
                    # Identify regimes in test set using the trained threshold
                    test_regime_vals = X_test[regime_col]
                    low_vol_mask = test_regime_vals <= model.regime_threshold
                    high_vol_mask = ~low_vol_mask
                    
                    # Low Vol Metrics
                    if low_vol_mask.sum() > 5:
                        y_test_low = y_test_arr[low_vol_mask]
                        y_pred_low = y_pred_arr[low_vol_mask]
                        low_ic = metrics.calculate_ic(y_test_low, y_pred_low)
                        low_decile = metrics.calculate_decile_analysis(y_test_low, y_pred_low)
                        fold_entry['low_vol_ic'] = low_ic
                        fold_entry['low_vol_spread'] = low_decile['spread']
                        fold_entry['low_vol_count'] = int(low_vol_mask.sum())
                    else:
                        fold_entry['low_vol_ic'] = np.nan
                        fold_entry['low_vol_spread'] = np.nan
                        fold_entry['low_vol_count'] = 0
                        
                    # High Vol Metrics
                    if high_vol_mask.sum() > 5:
                        y_test_high = y_test_arr[high_vol_mask]
                        y_pred_high = y_pred_arr[high_vol_mask]
                        high_ic = metrics.calculate_ic(y_test_high, y_pred_high)
                        high_decile = metrics.calculate_decile_analysis(y_test_high, y_pred_high)
                        fold_entry['high_vol_ic'] = high_ic
                        fold_entry['high_vol_spread'] = high_decile['spread']
                        fold_entry['high_vol_count'] = int(high_vol_mask.sum())
                    else:
                        fold_entry['high_vol_ic'] = np.nan
                        fold_entry['high_vol_spread'] = np.nan
                        fold_entry['high_vol_count'] = 0
                        
                    print(f"  Regime Split: Low Vol={fold_entry['low_vol_count']}, High Vol={fold_entry['high_vol_count']}")
                    print(f"  Low Vol IC: {fold_entry['low_vol_ic']:.4f}, Spread: {fold_entry['low_vol_spread']:.4f}")
                    print(f"  High Vol IC: {fold_entry['high_vol_ic']:.4f}, Spread: {fold_entry['high_vol_spread']:.4f}")
            
            # Add validation metrics if available
            if y_val_pred is not None and y_val_actual is not None and len(y_val_actual) > 0:
                val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
                val_mae = mean_absolute_error(y_val_actual, y_val_pred)
                val_dir_acc = np.mean(np.sign(y_val_actual) == np.sign(y_val_pred)) * 100
                val_ic = metrics.calculate_ic(y_val_actual, y_val_pred)
                val_tail = metrics.calculate_tail_metrics(y_val_actual, y_val_pred, threshold=big_move_thresh)
                
                fold_entry.update({
                    'n_val': len(y_val_actual),
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_dir_acc': val_dir_acc,
                    'val_ic': val_ic,
                    'val_big_up_precision': val_tail['precision_up_strict'],
                    'val_big_up_recall': val_tail['recall_up_strict'],
                    'val_big_down_precision': val_tail['precision_down_strict'],
                    'val_big_down_recall': val_tail['recall_down_strict']
                })
                
                # --- Per-Fold Threshold Tuning (Anti-Policy-Overfit) ---
                if config.WF_TUNE_THRESHOLD:
                    # Get val dates for monthly execution mode
                    val_dates = None
                    if execution_frequency == "monthly":
                        try:
                            # Try to get dates from y_val_orig index first (most reliable)
                            if hasattr(y_val_orig, 'index') and len(y_val_orig.index) == len(y_val_actual):
                                val_dates = pd.DatetimeIndex(y_val_orig.index)
                            # Otherwise try to get from df using val_idx
                            elif hasattr(val_idx, '__len__') and len(val_idx) > 0:
                                idx_list = list(val_idx) if not isinstance(val_idx, (list, np.ndarray)) else val_idx
                                val_dates = pd.DatetimeIndex(df.index[idx_list])
                        except Exception as e:
                            print(f"  Warning: Could not collect val dates for fold {fold}: {e}")
                            val_dates = None
                    
                    tuned_result = metrics.tune_and_evaluate_fold(
                        y_val_true=y_val_actual,
                        y_val_pred=y_val_pred,
                        y_test_true=y_test_arr,
                        y_test_pred=y_pred_arr,
                        criterion=config.WF_THRESHOLD_CRITERION,
                        n_grid_points=config.WF_THRESHOLD_N_GRID,
                        min_trade_fraction=config.WF_THRESHOLD_MIN_TRADE_FRAC,
                        frequency=frequency,
                        apply_vol_targeting=config.WF_THRESHOLD_VOL_TARGETING,
                        val_dates=val_dates,
                        test_dates=fold_test_dates,
                        execution_frequency=execution_frequency
                    )
                    
                    # Log the tuned threshold for this fold
                    tuned_tau = tuned_result['tuned_threshold']
                    tuned_test_sharpe = tuned_result['test_metrics']['sharpe']
                    print(f"  [Threshold Tuning] τ={tuned_tau:.4f} → Test Sharpe={tuned_test_sharpe:.2f}")
                    
                    # Add to fold entry
                    fold_entry.update({
                        'tuned_threshold': tuned_tau,
                        'tuned_val_sharpe': tuned_result['val_metrics']['sharpe'],
                        'tuned_test_sharpe': tuned_test_sharpe,
                        'tuned_test_return': tuned_result['test_metrics']['total_return'],
                        'tuned_test_hit_rate': tuned_result['test_metrics']['hit_rate'],
                        'tuned_test_trades': tuned_result['test_metrics']['trade_count']
                    })
                    
                    threshold_tuning_results.append(tuned_result)
            
            fold_metrics_list.append(fold_entry)
        
        if fold % 5 == 0:
            print(f"Fold {fold}: Predicted {len(y_test)} samples. (Test Date: {test_idx.min().date()})")
            
    # 3. Aggregate Evaluation
    all_actuals = np.array(all_actuals)
    all_preds = np.array(all_preds)
    
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    mae = mean_absolute_error(all_actuals, all_preds)
    dir_acc = np.mean(np.sign(all_actuals) == np.sign(all_preds)) * 100
    
    # Advanced Metrics
    big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
    pred_clip = getattr(config, 'PRED_CLIP', None)
    execution_frequency = getattr(config, 'EXECUTION_FREQUENCY', 'daily')
    ic = metrics.calculate_ic(all_actuals, all_preds)
    
    # Prepare dates for monthly execution mode
    test_dates = None
    if execution_frequency == "monthly" and len(all_test_dates) > 0:
        import pandas as pd
        if isinstance(all_test_dates[0], pd.Timestamp):
            test_dates = pd.DatetimeIndex(all_test_dates)
        else:
            test_dates = pd.to_datetime(all_test_dates)
    
    strat_metrics = metrics.calculate_strategy_metrics(
        all_actuals, all_preds, pred_clip=pred_clip, 
        dates=test_dates, execution_frequency=execution_frequency
    )
    tail_metrics = metrics.calculate_tail_metrics(all_actuals, all_preds, threshold=big_move_thresh)
    bigmove_strat = metrics.calculate_bigmove_strategy_metrics(
        all_actuals, all_preds, threshold=big_move_thresh, pred_clip=pred_clip,
        dates=test_dates, execution_frequency=execution_frequency
    )
    
    print(f"\n--- Walk-Forward Results (Aggregated) ---")
    print(f"Total Samples: {len(all_actuals)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
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
    
    # --- Signal Concentration Analysis (Aggregated) ---
    signal_concentration = metrics.calculate_signal_concentration(
        all_actuals, all_preds, frequency=frequency,
        dates=test_dates, execution_frequency=execution_frequency
    )
    metrics.print_signal_concentration_report(signal_concentration, title="Signal Concentration Analysis (Aggregated)")
    
    # --- Per-Fold Signal Quality Summary ---
    if len(fold_metrics_list) > 0:
        fold_ics = [f['ic'] for f in fold_metrics_list if 'ic' in f]
        fold_spreads = [f['decile_spread'] for f in fold_metrics_list if 'decile_spread' in f]
        fold_tstats = [f['decile_spread_tstat'] for f in fold_metrics_list if 'decile_spread_tstat' in f]
        fold_monos = [f['decile_monotonicity'] for f in fold_metrics_list if 'decile_monotonicity' in f]
        fold_best_sharpes = [f['best_thresh_sharpe'] for f in fold_metrics_list if 'best_thresh_sharpe' in f]
        
        print(f"\n{'='*70}")
        print(f" Per-Fold Signal Quality Summary ({len(fold_metrics_list)} folds)")
        print(f"{'='*70}")
        print(f"\n  IC across folds:           {np.mean(fold_ics):.4f} ± {np.std(fold_ics):.4f}")
        if fold_spreads:
            print(f"  Decile Spread (avg):       {np.mean(fold_spreads):+.4f} ± {np.std(fold_spreads):.4f}")
            print(f"  Decile T-stat (avg):       {np.mean(fold_tstats):+.2f}")
            print(f"  Monotonicity (avg):        {np.mean(fold_monos):+.3f}")
        if fold_best_sharpes:
            print(f"  Best Thresh Sharpe (avg):  {np.mean(fold_best_sharpes):.2f} ± {np.std(fold_best_sharpes):.2f}")
        
        # Correlation: IC vs decile spread (should be positive)
        if len(fold_ics) == len(fold_spreads) and len(fold_ics) > 3:
            from scipy.stats import spearmanr
            ic_spread_corr, _ = spearmanr(fold_ics, fold_spreads)
            print(f"  IC-Spread correlation:     {ic_spread_corr:+.3f}")
        
        print(f"{'='*70}")
    
    # --- Threshold Tuning Aggregated Results ---
    tuned_policy_agg = None
    if config.WF_TUNE_THRESHOLD and len(threshold_tuning_results) > 0:
        tuned_policy_agg = metrics.aggregate_tuned_policy_results(threshold_tuning_results)
        metrics.print_threshold_tuning_summary(
            tuned_policy_agg, 
            title=f"Threshold-Tuned Thresholded Policy (criterion={config.WF_THRESHOLD_CRITERION})"
        )
    
    # 4. Plot - Test Set (keep original filename for backward compatibility)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(all_actuals, label='Actual', alpha=0.7)
    plt.plot(all_preds, label='Predicted (Walk-Forward)', alpha=0.7)
    plt.title(f"Walk-Forward Test Set: {config.MODEL_TYPE}")
    plt.legend()
    plt.grid(True)
    logger.save_plot(fig, filename="forecast_plot_walkforward.png")
    
    # Also save with explicit test suffix
    logger.save_plot(fig, filename="forecast_plot_walkforward_test.png")
    
    # Plot - Train Set
    if len(all_train_preds) > 0 and len(all_train_actuals) > 0:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(all_train_actuals, label='Actual', alpha=0.7)
        plt.plot(all_train_preds, label='Predicted', alpha=0.7)
        plt.title(f"Walk-Forward Train Set: {config.MODEL_TYPE}")
        plt.legend()
        plt.grid(True)
        logger.save_plot(fig, filename="forecast_plot_walkforward_train.png")
    
    # Plot - Validation Set
    if len(all_val_preds) > 0 and len(all_val_actuals) > 0:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(all_val_actuals, label='Actual', alpha=0.7)
        plt.plot(all_val_preds, label='Predicted', alpha=0.7)
        plt.title(f"Walk-Forward Validation Set: {config.MODEL_TYPE}")
        plt.legend()
        plt.grid(True)
        logger.save_plot(fig, filename="forecast_plot_walkforward_val.png")
    
    # Construct metrics dicts for logger
    metrics_test = {
        'rmse': rmse, 
        'mae': mae, 
        'dir_acc': dir_acc,
        'ic': ic,
        'strat_metrics': strat_metrics,
        'tail_metrics': tail_metrics,
        'bigmove_strat': bigmove_strat,
        'signal_concentration': signal_concentration
    }
    
    # Aggregate validation metrics from folds if available
    folds_with_val = [f for f in fold_metrics_list if 'val_rmse' in f]
    if len(folds_with_val) > 0:
        avg_val_rmse = np.mean([f['val_rmse'] for f in folds_with_val])
        avg_val_mae = np.mean([f['val_mae'] for f in folds_with_val])
        avg_val_dir_acc = np.mean([f['val_dir_acc'] for f in folds_with_val])
        avg_val_ic = np.mean([f['val_ic'] for f in folds_with_val])
        
        metrics_val = {
            'rmse': avg_val_rmse,
            'mae': avg_val_mae,
            'dir_acc': avg_val_dir_acc,
            'ic': avg_val_ic
        }
        print(f"\n--- Validation Metrics (Averaged over {len(folds_with_val)} folds) ---")
        print(f"RMSE: {avg_val_rmse:.6f}")
        print(f"MAE:  {avg_val_mae:.6f}")
        print(f"Directional Accuracy: {avg_val_dir_acc:.2f}%")
        print(f"IC: {avg_val_ic:.4f}")
    else:
        # No validation data available (WF_VAL_MONTHS == 0 or merged into train)
        metrics_val = None
    
    metrics_train = None  # Walk-forward doesn't compute train metrics on final model
    
    # Save fold-level metrics to CSV
    logger.save_fold_metrics_csv(fold_metrics_list)
    
    # Add threshold tuning info to metrics if available
    if tuned_policy_agg is not None:
        metrics_test['threshold_tuning'] = {
            'enabled': True,
            'criterion': config.WF_THRESHOLD_CRITERION,
            'threshold_mean': tuned_policy_agg['threshold_mean'],
            'threshold_std': tuned_policy_agg['threshold_std'],
            'threshold_range': [tuned_policy_agg['threshold_min'], tuned_policy_agg['threshold_max']],
            'thresholds_per_fold': tuned_policy_agg['thresholds_per_fold'],
            'val_sharpe_mean': tuned_policy_agg['val_sharpe_mean'],
            'test_sharpe_mean': tuned_policy_agg['test_sharpe_mean'],
            'test_sharpe_std': tuned_policy_agg['test_sharpe_std'],
            'test_hit_rate_mean': tuned_policy_agg['test_hit_rate_mean'],
            'test_ic_mean': tuned_policy_agg['test_ic_mean'],
            'test_trade_count_total': tuned_policy_agg['test_trade_count_total']
        }
    else:
        metrics_test['threshold_tuning'] = {'enabled': False}
    
    # Save configuration JSON for reproducibility
    logger.save_config_json(config.MODEL_TYPE, best_params)
    
    # Log summary with diagnostics
    logger.log_summary(
        metrics_train, metrics_val, metrics_test, 
        config.MODEL_TYPE, feature_cols,
        y_true=all_actuals, 
        y_pred=all_preds,
        target_scaling_info=None,  # Walk-forward uses per-fold scaling, so no single value
        fold_metrics=fold_metrics_list
    )

    # --- PREPARE DATA FOR BOSS REPORT ---
    print("\nGenerating Boss Report...")
    
    # 1. Reload data to ensure we have the raw daily returns
    full_data_w_returns = data_prep.load_and_prep_data() 
    
    # 2. Ensure 'Return_1D' exists (Calculated from Price if missing)
    if 'Return_1D' not in full_data_w_returns.columns:
        if 'Adj Close' in full_data_w_returns.columns:
            full_data_w_returns['Return_1D'] = full_data_w_returns['Adj Close'].pct_change()
        else:
            print("Warning: 'Adj Close' not found. Approximating daily returns from target.")
            full_data_w_returns['Return_1D'] = full_data_w_returns[config.TARGET_COL] / 21.0 

    # 3. Align dates and returns
    test_dates = pd.DatetimeIndex(all_test_dates) 
    
    # We need to match the length of all_preds
    if len(test_dates) != len(all_preds):
        print(f"Warning: Date count ({len(test_dates)}) != Prediction count ({len(all_preds)}). Truncating to minimum.")
        min_len = min(len(test_dates), len(all_preds))
        test_dates = test_dates[:min_len]
        all_preds = all_preds[:min_len]

    aligned_daily_rets = full_data_w_returns.loc[test_dates, 'Return_1D']
    
    # 4. Run the Backtest Engine
    from ML.backtest_engine import BacktestEngine
    
    engine = BacktestEngine(
        predictions=all_preds,
        dates=test_dates,
        daily_returns=aligned_daily_rets,
        target_horizon=21
    )
    
    boss_report = engine.generate_boss_report_md()
    
    # 5. Append to Summary & Print
    print(boss_report)
    
    summary_path = os.path.join(logger.run_dir, "summary.md")
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(boss_report)
    
    print(f"\n✅ Boss Report appended to: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type to train')
    args = parser.parse_args()
    
    if args.model_type:
        config.MODEL_TYPE = args.model_type
        
    main()
