import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils

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
        step_months=1
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
        model = models.ModelFactory.get_model(config.MODEL_TYPE)
        model.fit(X_train, y_train)
        
        # Predict
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
    
    print(f"\n--- Walk-Forward Results (Aggregated) ---")
    print(f"Total Samples: {len(all_actuals)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    
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
    metrics_test = {'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
    metrics_val = {'rmse': 0, 'mae': 0, 'dir_acc': 0} # Placeholder
    metrics_train = {'rmse': 0, 'mae': 0, 'dir_acc': 0} # Placeholder
    
    logger.log_summary(metrics_train, metrics_val, metrics_test, config.MODEL_TYPE, feature_cols)

if __name__ == "__main__":
    main()
