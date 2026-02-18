import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ML import config, data_prep, utils, models, lstm_dataset


def train_ensemble():
    print("\n=== ASSEMBLING THE ALPHA ENGINE ===")
    print("Strategy: Ensemble (LSTM + Transformer + Linear) with Volatility Targeting")
    
    # Initialize Logger
    logger = utils.ExperimentLogger(model_name="Ensemble", process_tag="WalkForward", loss_tag="MSE")
    
    # 1. Load Data (need raw data for daily returns)
    print(f"Loading data from {config.DATA_PATH}...")
    df_raw = pd.read_csv(config.DATA_PATH, index_col=0, parse_dates=True)
    df_raw.sort_index(inplace=True)
    
    # Calculate DAILY returns for strategy backtesting BEFORE data_prep drops SPY_Price
    if 'SPY_Price' in df_raw.columns:
        df_raw['Daily_Return'] = df_raw['SPY_Price'].pct_change()
    else:
        raise ValueError("SPY_Price needed for daily returns calculation")
    
    # Load prepped data using dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    # Merge daily returns back (they're not in prepped data)
    df['Daily_Return'] = df_raw.loc[df.index, 'Daily_Return']
    
    # 2. Setup Walk-Forward Splitter with frequency-aware configuration
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,
        embargo_rows=embargo_rows,
        step_months=1,
        train_start_date=config.TRAIN_START_DATE,  # Use expanding window like train_walkforward.py
        frequency=frequency
    )
    
    target_col = config.TARGET_COL
    feature_cols = [c for c in df.columns if c not in [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown', 'Daily_Return']]
    
    # Storage
    predictions = {
        'Linear': [],
        'LSTM': [],
        'Transformer': [],
        'Ensemble': []
    }
    actuals = []
    dates = []
    
    # --- MODEL WEIGHTS ---
    # LSTM gets highest weight due to superior Sharpe/IC
    W_LSTM = 0.4
    W_TRANS = 0.3
    W_LIN = 0.3
    
    # Materialize splits for progress tracking
    splits = list(splitter.split(df))
    total_folds = len(splits)
    
    for fold, train_idx, val_idx, test_idx in splits:
        print(f"\n=== Fold {fold+1}/{total_folds} ===")
        
        # Data Slicing
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, target_col]
        
        # Merge val into train if configured (consistent with train_walkforward.py)
        if not config.WF_TRAIN_ON_TRAIN_PLUS_VAL and len(val_idx) > 0:
            X_val = df.loc[val_idx, feature_cols]
            y_val = df.loc[val_idx, target_col]
        else:
            # Merge train+val for training
            if len(val_idx) > 0:
                X_train = pd.concat([X_train, df.loc[val_idx, feature_cols]])
                y_train = pd.concat([y_train, df.loc[val_idx, target_col]])
            X_val, y_val = None, None
        
        # Scaling
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_test_s = scaler_x.transform(X_test)
        
        # Target Scaling (Standardize)
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-8:
            y_std = 1.0
        y_train_s = (y_train - y_mean) / y_std
        
        test_len = len(X_test)
        
        # --- 1. Linear Regression ---
        model_lin = LinearRegression()
        model_lin.fit(X_train_s, y_train)  # Linear trains on unscaled y
        pred_lin = model_lin.predict(X_test_s)
        
        # --- Deep Learning Prep ---
        time_steps_lstm = getattr(config, 'LSTM_TIME_STEPS', 10)
        time_steps_trans = getattr(config, 'TRANSFORMER_TIME_STEPS', 20)
        
        # Initialize prediction arrays with NaN (safer than padding with first prediction)
        pred_lstm_full = np.full(test_len, np.nan)
        pred_trans_full = np.full(test_len, np.nan)
        
        # --- 2. LSTM ---
        X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_s, y_train_s.values, time_steps_lstm)
        X_test_seq, _ = lstm_dataset.create_sequences(X_test_s, np.zeros(len(X_test_s)), time_steps_lstm)
        
        if len(X_train_seq) > 0 and len(X_test_seq) > 0:
            input_dim = X_train_seq.shape[2]
            
            # Use tuned params if available, else config defaults
            hidden_dim = getattr(config, 'LSTM_HIDDEN_DIM', 32)
            num_layers = getattr(config, 'LSTM_LAYERS', 1)
            dropout = 0.2
            
            from ML.models import LSTMModel
            model_lstm = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1, dropout=dropout)
            optimizer = torch.optim.Adam(model_lstm.parameters(), lr=config.LSTM_LEARNING_RATE)
            
            # Training with gradient clipping
            model_lstm.train()
            train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=config.LSTM_BATCH_SIZE)
            
            epochs = config.LSTM_EPOCHS
            for epoch in range(epochs):
                epoch_loss = 0.0
                for X_b, y_b in train_loader:
                    optimizer.zero_grad()
                    out = model_lstm(X_b)
                    loss = utils.compute_loss(out, y_b, loss_mode="mse")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_loss = epoch_loss / max(1, len(train_loader))
                    print(f"  LSTM Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            # Predict
            model_lstm.eval()
            with torch.no_grad():
                pred_lstm_s = model_lstm(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
            
            # Unscale & place in correct position (sequences start at index time_steps-1)
            pred_lstm = pred_lstm_s * y_std + y_mean
            pred_lstm_full[time_steps_lstm-1:time_steps_lstm-1+len(pred_lstm)] = pred_lstm
        else:
            print(f"  LSTM: Insufficient data for sequences")

        # --- 3. Transformer ---
        X_train_seq_t, y_train_seq_t = lstm_dataset.create_sequences(X_train_s, y_train_s.values, time_steps_trans)
        X_test_seq_t, _ = lstm_dataset.create_sequences(X_test_s, np.zeros(len(X_test_s)), time_steps_trans)
        
        if len(X_train_seq_t) > 0 and len(X_test_seq_t) > 0:
            input_dim = X_train_seq_t.shape[2]
            
            # Use config params for Transformer
            model_dim = getattr(config, 'TRANSFORMER_MODEL_DIM', 64)
            num_heads = getattr(config, 'TRANSFORMER_HEADS', 4)
            num_layers = getattr(config, 'TRANSFORMER_LAYERS', 2)
            dim_feedforward = getattr(config, 'TRANSFORMER_FEEDFORWARD_DIM', 128)
            dropout = getattr(config, 'TRANSFORMER_DROPOUT', 0.1)
            
            from ML.transformer.model import TransformerModel
            model_trans = TransformerModel(
                input_dim=input_dim,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            
            # Optimizer with weight decay for regularization
            optimizer = torch.optim.Adam(
                model_trans.parameters(), 
                lr=config.TRANSFORMER_LR,
                weight_decay=getattr(config, 'TRANSFORMER_WEIGHT_DECAY', 1e-4)
            )
            
            model_trans.train()
            train_loader_t = lstm_dataset.prepare_dataloader(X_train_seq_t, y_train_seq_t, batch_size=config.TRANSFORMER_BATCH_SIZE)
            
            epochs = config.TRANSFORMER_EPOCHS
            for epoch in range(epochs):
                epoch_loss = 0.0
                for X_b, y_b in train_loader_t:
                    optimizer.zero_grad()
                    out = model_trans(X_b)
                    loss = utils.compute_loss(out, y_b, loss_mode="mse")
                    loss.backward()
                    # Gradient clipping is critical for Transformer stability
                    torch.nn.utils.clip_grad_norm_(model_trans.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_loss = epoch_loss / max(1, len(train_loader_t))
                    print(f"  Transformer Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            model_trans.eval()
            with torch.no_grad():
                pred_trans_s = model_trans(torch.tensor(X_test_seq_t, dtype=torch.float32)).numpy().flatten()
            
            # Unscale & place in correct position
            pred_trans = pred_trans_s * y_std + y_mean
            pred_trans_full[time_steps_trans-1:time_steps_trans-1+len(pred_trans)] = pred_trans
        else:
            print(f"  Transformer: Insufficient data for sequences")

        # --- Blend (handle NaN gracefully) ---
        # For rows where deep models have NaN, fall back to Linear only
        pred_ens = np.zeros(test_len)
        for i in range(test_len):
            valid_preds = []
            weights = []
            
            # Linear is always valid
            valid_preds.append(pred_lin[i])
            weights.append(W_LIN)
            
            if not np.isnan(pred_lstm_full[i]):
                valid_preds.append(pred_lstm_full[i])
                weights.append(W_LSTM)
            
            if not np.isnan(pred_trans_full[i]):
                valid_preds.append(pred_trans_full[i])
                weights.append(W_TRANS)
            
            # Normalize weights and compute weighted average
            total_weight = sum(weights)
            pred_ens[i] = sum(p * w / total_weight for p, w in zip(valid_preds, weights))
        
        predictions['Linear'].extend(pred_lin)
        predictions['LSTM'].extend(pred_lstm_full)
        predictions['Transformer'].extend(pred_trans_full)
        predictions['Ensemble'].extend(pred_ens)
        actuals.extend(y_test.values)
        dates.extend(df.loc[test_idx].index)

    # --- RESULTS & RISK OVERLAY ---
    res_df = pd.DataFrame(predictions, index=dates)
    res_df['Actual'] = actuals  # 1-month forward return (for directional accuracy)
    
    # Merge daily returns for strategy calculation
    res_df['Daily_Return'] = df.loc[res_df.index, 'Daily_Return']
    
    # 1. Calc Raw Performance (exclude NaN for individual model metrics)
    valid_mask = ~np.isnan(res_df['Ensemble'])
    dir_acc = np.mean(np.sign(res_df.loc[valid_mask, 'Ensemble']) == np.sign(res_df.loc[valid_mask, 'Actual']))
    ic = res_df.loc[valid_mask, ['Ensemble', 'Actual']].corr().iloc[0, 1]
    
    # Individual model metrics
    lstm_valid = ~np.isnan(res_df['LSTM'])
    trans_valid = ~np.isnan(res_df['Transformer'])
    
    lstm_dir_acc = np.mean(np.sign(res_df.loc[lstm_valid, 'LSTM']) == np.sign(res_df.loc[lstm_valid, 'Actual'])) if lstm_valid.sum() > 0 else 0
    trans_dir_acc = np.mean(np.sign(res_df.loc[trans_valid, 'Transformer']) == np.sign(res_df.loc[trans_valid, 'Actual'])) if trans_valid.sum() > 0 else 0
    lin_dir_acc = np.mean(np.sign(res_df['Linear']) == np.sign(res_df['Actual']))
    
    print(f"\n\n=== ENSEMBLE RESULTS ===")
    print(f"Total Predictions: {len(res_df)}")
    print(f"\nIndividual Model Directional Accuracy:")
    print(f"  Linear:      {lin_dir_acc*100:.2f}%")
    print(f"  LSTM:        {lstm_dir_acc*100:.2f}% (valid: {lstm_valid.sum()})")
    print(f"  Transformer: {trans_dir_acc*100:.2f}% (valid: {trans_valid.sum()})")
    print(f"\nEnsemble:")
    print(f"  Directional Accuracy: {dir_acc*100:.2f}%")
    print(f"  Information Coefficient (IC): {ic:.4f}")
    
    # 2. Apply Volatility Targeting (The Risk Engine)
    # Target 15% Volatility (annualized)
    TARGET_VOL = 0.15
    # Calculate rolling realized vol using DAILY returns (63 trading days = ~3 months)
    # Annualize by sqrt(252)
    rolling_vol = res_df['Daily_Return'].rolling(63, min_periods=21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.fillna(0.15)  # Default to target vol
    
    # Vol Scalar: If market vol > 15%, scale down. If < 15%, scale up (max 2.0).
    vol_scalar = (TARGET_VOL / rolling_vol).clip(0.5, 2.0).shift(1).fillna(1.0)
    
    # Strategy Returns using DAILY returns (not overlapping monthly!)
    res_df['Signal_Raw'] = np.sign(res_df['Ensemble'])
    res_df['Ret_Raw'] = res_df['Signal_Raw'] * res_df['Daily_Return']
    res_df['Ret_VolTarget'] = res_df['Ret_Raw'] * vol_scalar
    
    # Metrics using DAILY returns (annualize by sqrt(252))
    cum_raw = (1 + res_df['Ret_Raw']).cumprod()
    cum_vol = (1 + res_df['Ret_VolTarget']).cumprod()
    cum_bh = (1 + res_df['Daily_Return']).cumprod()  # Buy & Hold uses daily returns too
    
    sharpe_raw = res_df['Ret_Raw'].mean() / res_df['Ret_Raw'].std() * np.sqrt(252) if res_df['Ret_Raw'].std() > 0 else 0
    sharpe_vol = res_df['Ret_VolTarget'].mean() / res_df['Ret_VolTarget'].std() * np.sqrt(252) if res_df['Ret_VolTarget'].std() > 0 else 0
    sharpe_bh = res_df['Daily_Return'].mean() / res_df['Daily_Return'].std() * np.sqrt(252) if res_df['Daily_Return'].std() > 0 else 0
    
    dd_raw = (cum_raw / cum_raw.cummax() - 1).min()
    dd_vol = (cum_vol / cum_vol.cummax() - 1).min()
    dd_bh = (cum_bh / cum_bh.cummax() - 1).min()
    
    total_ret_raw = cum_raw.iloc[-1] - 1
    total_ret_vol = cum_vol.iloc[-1] - 1
    total_ret_bh = cum_bh.iloc[-1] - 1
    
    print(f"\n=== STRATEGY METRICS ===")
    print(f"{'Strategy':<25} {'Sharpe':>10} {'Total Ret':>12} {'Max DD':>10}")
    print(f"{'-'*60}")
    print(f"{'Buy & Hold':<25} {sharpe_bh:>10.2f} {total_ret_bh:>11.1%} {dd_bh:>10.1%}")
    print(f"{'Ensemble (Raw)':<25} {sharpe_raw:>10.2f} {total_ret_raw:>11.1%} {dd_raw:>10.1%}")
    print(f"{'Ensemble (Vol Target 15%)':<25} {sharpe_vol:>10.2f} {total_ret_vol:>11.1%} {dd_vol:>10.1%}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Cumulative Returns
    ax1 = axes[0]
    ax1.plot(cum_bh.index, cum_bh.values, label='Buy & Hold', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(cum_raw.index, cum_raw.values, label='Ensemble (Raw)', alpha=0.7, linewidth=1.5)
    ax1.plot(cum_vol.index, cum_vol.values, label='Ensemble (Vol Targeted)', linewidth=2)
    ax1.set_title("Ensemble Strategy: Cumulative Returns", fontsize=14)
    ax1.set_ylabel("Cumulative Return (Growth of $1)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    
    # Bottom: Rolling Sharpe (63 trading days = ~3 months)
    ax2 = axes[1]
    roll_sharpe_ens = res_df['Ret_VolTarget'].rolling(63).mean() / res_df['Ret_VolTarget'].rolling(63).std() * np.sqrt(252)
    roll_sharpe_bh = res_df['Daily_Return'].rolling(63).mean() / res_df['Daily_Return'].rolling(63).std() * np.sqrt(252)
    ax2.plot(roll_sharpe_bh.index, roll_sharpe_bh.values, label='Buy & Hold', linestyle='--', alpha=0.6)
    ax2.plot(roll_sharpe_ens.index, roll_sharpe_ens.values, label='Ensemble (Vol Targeted)', linewidth=1.5)
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Sharpe = 1')
    ax2.set_title("Rolling 3-Month Sharpe Ratio", fontsize=14)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 5)
    
    plt.tight_layout()
    
    # Save using logger
    logger.save_plot(fig, filename="ensemble_performance.png")
    
    # Save predictions to CSV
    res_df.to_csv(f"{logger.run_dir}/ensemble_predictions.csv")
    print(f"\nSaved predictions to {logger.run_dir}/ensemble_predictions.csv")
    
    # Log summary
    summary_text = f"""# Ensemble Strategy Summary

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Weights
- Linear: {W_LIN:.0%}
- LSTM: {W_LSTM:.0%}
- Transformer: {W_TRANS:.0%}

## Configuration
- Test Start: {config.TEST_START_DATE}
- Train Window: {config.TRAIN_WINDOW_YEARS} years
- Val Window: {config.WF_VAL_MONTHS} months
- Target Volatility: {TARGET_VOL:.0%}

## Individual Model Performance
| Model | Directional Accuracy | Valid Predictions |
|-------|---------------------|-------------------|
| Linear | {lin_dir_acc*100:.2f}% | {len(res_df)} |
| LSTM | {lstm_dir_acc*100:.2f}% | {lstm_valid.sum()} |
| Transformer | {trans_dir_acc*100:.2f}% | {trans_valid.sum()} |


## Ensemble Performance
- Directional Accuracy: {dir_acc*100:.2f}%
- Information Coefficient: {ic:.4f}

## Strategy Metrics
| Strategy | Sharpe | Total Return | Max Drawdown |
|----------|--------|--------------|--------------|
| Buy & Hold | {sharpe_bh:.2f} | {total_ret_bh:.1%} | {dd_bh:.1%} |
| Ensemble (Raw) | {sharpe_raw:.2f} | {total_ret_raw:.1%} | {dd_raw:.1%} |
| Ensemble (Vol Target) | {sharpe_vol:.2f} | {total_ret_vol:.1%} | {dd_vol:.1%} |
"""
    
    with open(f"{logger.run_dir}/summary.md", 'w') as f:
        f.write(summary_text)
    
    print(f"\nResults saved to: {logger.run_dir}")
    
    return res_df


if __name__ == "__main__":
    train_ensemble()
