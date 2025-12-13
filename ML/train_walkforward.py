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
        
        # Merge train+val if configured and val is non-empty
        if config.WF_TRAIN_ON_TRAIN_PLUS_VAL and len(val_idx) > 0:
            X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col]
            print(f"  Merging train ({len(X_train)}) + val ({len(X_val)}) for training")
            X_train = pd.concat([X_train, X_val])
            y_train = pd.concat([y_train, y_val])
        
        # Train
        if config.MODEL_TYPE in ['LSTM', 'CNN', 'Transformer']:
            # --- Deep Learning Training Logic ---
            # 1. Standardize target per fold for stability
            y_mean = y_train.mean()
            y_std = y_train.std()
            if y_std == 0:
                y_std = 1.0
            y_train_scaled = (y_train - y_mean) / y_std
            y_test_scaled = (y_test - y_mean) / y_std
            
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
            
            # 6. Train Loop
            model.train()
            for epoch in range(epochs):
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
                    # Gradient clipping for stability (especially important for Transformer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.WF_GRAD_CLIP_NORM)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    epoch_loss += loss.item()
                
                # Lightweight progress logging
                if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
                    avg_loss = epoch_loss / max(1, len(train_loader))
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            # 7. Predict and Unscale
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                y_pred_scaled = model(X_test_tensor).numpy().flatten()
            
            # Unscale predictions back to original scale
            y_pred = y_pred_scaled * y_std + y_mean
                
            # Adjust Actuals (slice off first time_steps-1) - use raw y_test for metrics
            y_test = y_test.iloc[time_steps-1:]
            
        else:
            # --- Standard ML Training ---
            model = models.ModelFactory.get_model(config.MODEL_TYPE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Store
        all_preds.extend(y_pred)
        all_actuals.extend(y_test)
        
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
    ic = metrics.calculate_ic(all_actuals, all_preds)
    strat_metrics = metrics.calculate_strategy_metrics(all_actuals, all_preds)
    tail_metrics = metrics.calculate_tail_metrics(all_actuals, all_preds, threshold=big_move_thresh)
    bigmove_strat = metrics.calculate_bigmove_strategy_metrics(all_actuals, all_preds, threshold=big_move_thresh)
    
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
    metrics_val = {'rmse': 0, 'mae': 0, 'dir_acc': 0} # Placeholder
    metrics_train = {'rmse': 0, 'mae': 0, 'dir_acc': 0} # Placeholder
    
    logger.log_summary(metrics_train, metrics_val, metrics_test, config.MODEL_TYPE, feature_cols)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Model type to train')
    args = parser.parse_args()
    
    if args.model_type:
        config.MODEL_TYPE = args.model_type
        
    main()
