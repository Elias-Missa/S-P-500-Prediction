import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ML import config, data_prep, models, utils

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
    
    return {'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}

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
        buffer_days=config.BUFFER_DAYS
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
    model = models.ModelFactory.get_model(config.MODEL_TYPE)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    # Validation
    y_val_pred = model.predict(X_val)
    metrics_val = evaluate(y_val, y_val_pred, "Validation")
    
    # Test (Out of Sample)
    y_test_pred = model.predict(X_test)
    metrics_test = evaluate(y_test, y_test_pred, "Test (OOS)")
    
    # Dummy train metrics for logging (optional, or calculate real ones)
    metrics_train = {'rmse': 0, 'mae': 0, 'dir_acc': 0} 
    
    # 5. Plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Return', alpha=0.7)
    plt.plot(y_test.index, y_test_pred, label='Predicted Return', alpha=0.7)
    plt.title(f"Test Set: Actual vs Predicted ({config.MODEL_TYPE})")
    plt.legend()
    plt.grid(True)
    
    # Log Results
    logger.save_plot(fig, filename="forecast_plot.png")
    logger.log_summary(metrics_train, metrics_val, metrics_test, config.MODEL_TYPE, feature_cols)
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        print("\nTop 10 Features:")
        print(importances.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
