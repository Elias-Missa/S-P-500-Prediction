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
    strat_metrics = metrics.calculate_strategy_metrics(y_true, y_pred)
    tail_metrics = metrics.calculate_tail_metrics(y_true, y_pred, threshold=0.05)
    
    print(f"IC: {ic:.4f}")
    print(f"Sharpe (Ann.): {strat_metrics['sharpe']:.2f}")
    print(f"Big Shift Precision (Up): {tail_metrics['precision_up_strict']:.2f}")
    print(f"Big Shift Recall (Up): {tail_metrics['recall_up_strict']:.2f}")
    
    return {
        'rmse': rmse, 
        'mae': mae, 
        'dir_acc': dir_acc,
        'ic': ic,
        'strat_metrics': strat_metrics,
        'tail_metrics': tail_metrics
    }

def main():
    # Initialize Logger
    logger = utils.ExperimentLogger(model_name=config.MODEL_TYPE, process_tag="Static")

    # 1. Load Data
    df = data_prep.load_and_prep_data()
    
    # 2. Split Data
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        buffer_days=config.BUFFER_DAYS,
        train_start_date=config.TRAIN_START_DATE
    )
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    # Features and Target
    target_col = config.TARGET_COL
    feature_cols = [c for c in df.columns if c != target_col]
    
    X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
    X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col]
    X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, target_col]
    
    print(f"\nTraining on {len(X_train)} samples, Validating on {len(X_val)}, Testing on {len(X_test)}")
    
    # 3. Train Model
    print(f"Training {config.MODEL_TYPE}...")
    
    # --- Optuna Tuning ---
    best_params = {}
    if config.USE_OPTUNA:
        tuner = tuning.HyperparameterTuner(config.MODEL_TYPE, X_train, y_train, X_val, y_val)
        best_params = tuner.optimize(n_trials=config.OPTUNA_TRIALS)
        
        # Update config with best params (Temporary override for this run)
        if config.MODEL_TYPE == 'LSTM':
            config.LSTM_HIDDEN_DIM = best_params.get('hidden_dim', config.LSTM_HIDDEN_DIM)
            config.LSTM_LAYERS = best_params.get('num_layers', config.LSTM_LAYERS)
            config.LSTM_LEARNING_RATE = best_params.get('lr', config.LSTM_LEARNING_RATE)
            config.LSTM_BATCH_SIZE = best_params.get('batch_size', config.LSTM_BATCH_SIZE)
            # Dropout is passed directly to model init, not config global usually, but we can handle it
            
        elif config.MODEL_TYPE == 'XGBoost':
            config.XGB_N_ESTIMATORS = best_params.get('n_estimators', config.XGB_N_ESTIMATORS)
            config.XGB_LEARNING_RATE = best_params.get('learning_rate', config.XGB_LEARNING_RATE)
            # Update other globals if needed or rely on ModelFactory accepting kwargs (which we haven't implemented fully yet)
            # For now, we assume ModelFactory uses config globals.
            
        elif config.MODEL_TYPE == 'RandomForest':
            config.RF_N_ESTIMATORS = best_params.get('n_estimators', config.RF_N_ESTIMATORS)
            config.RF_MAX_DEPTH = best_params.get('max_depth', config.RF_MAX_DEPTH)
            config.RF_MIN_SAMPLES_SPLIT = best_params.get('min_samples_split', config.RF_MIN_SAMPLES_SPLIT)
            config.RF_MIN_SAMPLES_LEAF = best_params.get('min_samples_leaf', config.RF_MIN_SAMPLES_LEAF)
            
        elif config.MODEL_TYPE == 'MLP':
            # MLP params are tuples, need careful handling
            # Reconstruct hidden_layer_sizes tuple
            n_layers = best_params.get('n_layers', 2)
            layers = []
            for i in range(n_layers):
                layers.append(best_params.get(f'n_units_l{i}', 64))
            config.MLP_HIDDEN_LAYERS = tuple(layers)
            config.MLP_LEARNING_RATE_INIT = best_params.get('learning_rate_init', config.MLP_LEARNING_RATE_INIT)
            config.MLP_ALPHA = best_params.get('alpha', config.MLP_ALPHA)
            # We don't have a global for learning_rate_init in config yet, likely default.
            # We should probably update ModelFactory to accept these overrides.
            
        elif config.MODEL_TYPE == 'CNN':
            config.CNN_FILTERS = best_params.get('filters', config.CNN_FILTERS)
            config.CNN_KERNEL_SIZE = best_params.get('kernel_size', config.CNN_KERNEL_SIZE)
            config.CNN_LAYERS = best_params.get('layers', config.CNN_LAYERS)
            config.CNN_DROPOUT = best_params.get('dropout', config.CNN_DROPOUT)
            config.CNN_LEARNING_RATE = best_params.get('lr', config.CNN_LEARNING_RATE)
            config.CNN_BATCH_SIZE = best_params.get('batch_size', config.CNN_BATCH_SIZE)
            
    if config.MODEL_TYPE in ['LSTM', 'CNN']:
        # --- Deep Learning Training Logic ---
        print(f"Preparing data for {config.MODEL_TYPE}...")
        
        # 1. Scale Data (Fit on Train, Transform on Val/Test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. Reshape to 3D Sequences
        time_steps = config.LSTM_TIME_STEPS
        X_train_seq, y_train_seq = lstm_dataset.create_sequences(X_train_scaled, y_train.values, time_steps)
        X_val_seq, y_val_seq = lstm_dataset.create_sequences(X_val_scaled, y_val.values, time_steps)
        X_test_seq, y_test_seq = lstm_dataset.create_sequences(X_test_scaled, y_test.values, time_steps)
        
        # 3. Create DataLoaders
        train_loader = lstm_dataset.prepare_dataloader(X_train_seq, y_train_seq, batch_size=config.LSTM_BATCH_SIZE)
        
        # 4. Initialize Model
        input_dim = X_train_seq.shape[2]
        if config.MODEL_TYPE == 'LSTM':
            dropout = best_params.get('dropout', 0.2)
            model = models.LSTMModel(input_dim, config.LSTM_HIDDEN_DIM, config.LSTM_LAYERS, dropout=dropout)
            lr = config.LSTM_LEARNING_RATE
            epochs = config.LSTM_EPOCHS
        elif config.MODEL_TYPE == 'CNN':
            model = models.CNN1DModel(
                input_dim, 
                config.CNN_FILTERS, 
                config.CNN_KERNEL_SIZE, 
                config.CNN_LAYERS, 
                dropout=config.CNN_DROPOUT
            )
            lr = config.CNN_LEARNING_RATE
            epochs = config.CNN_EPOCHS
            
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 5. Train Loop
        print(f"Training {config.MODEL_TYPE} for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.LSTM_EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}")
                
        # 6. Predict (Evaluation)
        model.eval()
        with torch.no_grad():
            # Convert Train/Val/Test to tensor
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            y_train_pred = model(X_train_tensor).numpy().flatten()
            
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            y_val_pred = model(X_val_tensor).numpy().flatten()
            
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
            y_test_pred = model(X_test_tensor).numpy().flatten()
            
        # Adjust Actuals to match sequence length (first time_steps-1 are lost)
        y_train_actual = y_train.iloc[time_steps-1:]
        y_val_actual = y_val.iloc[time_steps-1:]
        y_test_actual = y_test.iloc[time_steps-1:]
        
        # Override for evaluation
        y_train = y_train_actual
        y_val = y_val_actual
        y_test = y_test_actual
        
    else:
        # --- Standard ML Training ---
        model = models.ModelFactory.get_model(config.MODEL_TYPE)
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
    
    # Log Results
    logger.log_summary(metrics_train, metrics_val, metrics_test, config.MODEL_TYPE, feature_cols)
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        print("\nTop 10 Features:")
        print(importances.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
