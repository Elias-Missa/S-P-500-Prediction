import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils, metrics, lstm_dataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def main():
    # Initialize Logger with loss mode tag
    loss_tag = config.LOSS_MODE.upper()
    logger = utils.ExperimentLogger(model_name=config.MODEL_TYPE, process_tag="WalkForward", loss_tag=loss_tag)
    
    print("--- Walk-Forward Validation ---")
    
    # 1. Load Data
    df = data_prep.load_and_prep_data()
    
    # 2. Setup Splitter
    # We start testing from TEST_START_DATE (2023-01-01)
    # We step forward by 1 month.
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,  # Use walk-forward specific val_months
        buffer_days=config.BUFFER_DAYS,
        step_months=1,
        train_start_date=config.TRAIN_START_DATE
    )
    
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
    
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Rolling Train Window: {config.TRAIN_WINDOW_YEARS} years")
    
    # Determine time_steps for deep models
    time_steps = None
    if config.MODEL_TYPE == 'LSTM':
        time_steps = getattr(config, 'LSTM_TIME_STEPS', 10)
    elif config.MODEL_TYPE == 'CNN':
        time_steps = getattr(config, 'CNN_TIME_STEPS', 10)
    elif config.MODEL_TYPE == 'Transformer':
        time_steps = getattr(config, 'TRANSFORMER_TIME_STEPS', 20)
    
    if time_steps is not None:
        print(f"Using time_steps={time_steps} for sequence creation")
    
    # Materialize splits so we can report progress and counts
    splits = list(splitter.split(df))
    total_folds = len(splits)
    
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
        
        # Train
        if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
            # --- Deep Learning Training Logic ---
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
            
            # 4. Create DataLoader
            if config.MODEL_TYPE == 'LSTM':
                batch_size = config.LSTM_BATCH_SIZE
            elif config.MODEL_TYPE == 'CNN':
                batch_size = config.CNN_BATCH_SIZE
            elif config.MODEL_TYPE == 'Transformer':
                batch_size = config.TRANSFORMER_BATCH_SIZE
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=batch_size)
            
            # 4b. Prepare Validation DataLoader for Early Stopping (if available)
            val_loader = None
            has_val = has_val_set and X_val_orig is not None and len(X_val_orig) >= time_steps
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
            
            # 5. Initialize Model
            input_dim = X_train_seq.shape[2]
            weight_decay = 0
            if config.MODEL_TYPE == 'LSTM':
                model = models.ModelFactory.get_model('LSTM', input_dim=input_dim)
                lr = config.LSTM_LEARNING_RATE
                epochs = config.LSTM_EPOCHS
            elif config.MODEL_TYPE == 'CNN':
                model = models.ModelFactory.get_model('CNN', input_dim=input_dim)
                lr = config.CNN_LEARNING_RATE
                epochs = config.CNN_EPOCHS
            elif config.MODEL_TYPE == 'Transformer':
                model = models.ModelFactory.get_model('Transformer', input_dim=input_dim)
                lr = config.TRANSFORMER_LR
                weight_decay = config.TRANSFORMER_WEIGHT_DECAY
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
                    if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
                        avg_loss = epoch_loss / max(1, len(train_loader))
                        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            # 3. Restore Best Weights (Critical!)
            if getattr(config, 'WF_EARLY_STOPPING', False) and best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"  Restored best model from validation (Loss: {best_val_loss:.6f})")
            
            # 7. Predict and Unscale
            model.eval()
            with torch.no_grad():
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
            model = models.ModelFactory.get_model(config.MODEL_TYPE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # --- Compute Validation Metrics for standard ML if val set is available ---
            y_val_pred, y_val_actual = None, None
            if has_val_set and X_val_orig is not None and len(X_val_orig) > 0:
                y_val_pred = model.predict(X_val_orig)
                y_val_actual = y_val_orig.values
        
        # Store
        all_preds.extend(y_pred)
        all_actuals.extend(y_test)
        
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
            fold_tail = metrics.calculate_tail_metrics(y_test_arr, y_pred_arr, threshold=big_move_thresh)
            fold_strat = metrics.calculate_strategy_metrics(y_test_arr, y_pred_arr, pred_clip=pred_clip)
            fold_bigmove_strat = metrics.calculate_bigmove_strategy_metrics(y_test_arr, y_pred_arr, threshold=big_move_thresh, pred_clip=pred_clip)
            
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
                'big_up_precision': fold_tail['precision_up_strict'],
                'big_up_recall': fold_tail['recall_up_strict'],
                'big_down_precision': fold_tail['precision_down_strict'],
                'big_down_recall': fold_tail['recall_down_strict'],
                'strategy_sharpe': fold_strat['sharpe'],
                'big_move_sharpe': fold_bigmove_strat['sharpe']
            }
            
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
    ic = metrics.calculate_ic(all_actuals, all_preds)
    strat_metrics = metrics.calculate_strategy_metrics(all_actuals, all_preds, pred_clip=pred_clip)
    tail_metrics = metrics.calculate_tail_metrics(all_actuals, all_preds, threshold=big_move_thresh)
    bigmove_strat = metrics.calculate_bigmove_strategy_metrics(all_actuals, all_preds, threshold=big_move_thresh, pred_clip=pred_clip)
    
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
    
    # 4. Plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(all_actuals, label='Actual', alpha=0.7)
    plt.plot(all_preds, label='Predicted (Walk-Forward)', alpha=0.7)
    plt.title(f"Walk-Forward Validation: {config.MODEL_TYPE}")
    plt.legend()
    plt.grid(True)
    
    # Log Results
    logger.save_plot(fig, filename="forecast_plot_walkforward.png")
    
    # Construct metrics dicts for logger
    metrics_test = {
        'rmse': rmse, 
        'mae': mae, 
        'dir_acc': dir_acc,
        'ic': ic,
        'strat_metrics': strat_metrics,
        'tail_metrics': tail_metrics,
        'bigmove_strat': bigmove_strat
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
    
    # Log summary with diagnostics
    logger.log_summary(
        metrics_train, metrics_val, metrics_test, 
        config.MODEL_TYPE, feature_cols,
        y_true=all_actuals, 
        y_pred=all_preds,
        target_scaling_info=None,  # Walk-forward uses per-fold scaling, so no single value
        fold_metrics=fold_metrics_list
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type to train')
    args = parser.parse_args()
    
    if args.model_type:
        config.MODEL_TYPE = args.model_type
        
    main()
