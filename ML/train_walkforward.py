import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils, metrics, lstm_dataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def main():
    # Initialize Logger
    logger = utils.ExperimentLogger(model_name=config.MODEL_TYPE, process_tag="WalkForward")
    
    print("--- Walk-Forward Validation ---")
    
    # 1. Load Data
    df = data_prep.load_and_prep_data()
    
    # 2. Setup Splitter
    # We start testing from TEST_START_DATE (2023-01-01)
    # We step forward by 1 month.
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        buffer_days=config.BUFFER_DAYS,
        step_months=1,
        train_start_date=config.TRAIN_START_DATE
    )
    
    target_col = config.TARGET_COL
    feature_cols = [c for c in df.columns if c != target_col]
    
    all_preds = []
    all_actuals = []
    
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Rolling Train Window: {config.TRAIN_WINDOW_YEARS} years")
    
    for fold, train_idx, val_idx, test_idx in splitter.split(df):
        # Prepare Data
        X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
        # X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col] # Can use for early stopping
        X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, target_col]
        
        # Train
        if config.MODEL_TYPE in ['LSTM', 'CNN']:
            # --- Deep Learning Training Logic ---
            # 1. Scale Data (Fit on Train, Transform on Test)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 2. Reshape to 3D Sequences
            time_steps = config.LSTM_TIME_STEPS
            X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, y_train.values, time_steps)
            X_test_seq, y_test_seq = lstm_dataset.create_sequences(X_test_scaled, y_test.values, time_steps)
            
            # Check if sequences are empty (e.g. if fold is too small)
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                print(f"Fold {fold}: Skipping due to insufficient data for sequence generation.")
                continue
            
            # 3. Create DataLoader
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=config.LSTM_BATCH_SIZE)
            
            # 4. Initialize Model
            input_dim = X_train_seq.shape[2]
            if config.MODEL_TYPE == 'LSTM':
                model = models.ModelFactory.get_model('LSTM', input_dim=input_dim)
                lr = config.LSTM_LEARNING_RATE
                epochs = config.LSTM_EPOCHS
            elif config.MODEL_TYPE == 'CNN':
                model = models.ModelFactory.get_model('CNN', input_dim=input_dim)
                lr = config.CNN_LEARNING_RATE
                epochs = config.CNN_EPOCHS
                
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # 5. Train Loop
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # 6. Predict
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                y_pred = model(X_test_tensor).numpy().flatten()
                
            # Adjust Actuals (slice off first time_steps-1)
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
