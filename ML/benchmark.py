import os

import pandas as pd

from ML import config, data_prep, models, tuning, feature_selection
from ML.train import evaluate, resolve_model_params


def run_benchmark():
    """Train and compare a suite of baseline models on a shared split."""
    # Load data using dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        embargo_rows=embargo_rows,
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    train_idx, val_idx, test_idx = splitter.get_split(df)

    target_col = config.TARGET_COL
    # Centralized feature selection
    feature_cols = feature_selection.select_feature_columns(df)

    X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
    X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target_col]
    X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, target_col]

    results = []
    summary_dir = os.path.join(config.REPO_ROOT, 'ML_Output')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'benchmark_summary.csv')

    for model_type in config.BASIC_MODEL_SUITE:
        print(f"\n===== Benchmarking {model_type} =====")
        best_params = {}
        if config.USE_OPTUNA:
            tuner = tuning.HyperparameterTuner(model_type, X_train, y_train, X_val, y_val)
            best_params = tuner.optimize(n_trials=config.OPTUNA_TRIALS)

        model_params = resolve_model_params(model_type, best_params)
        model = models.ModelFactory.get_model(model_type, overrides=model_params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        metrics_val = evaluate(y_val, y_val_pred, "Validation")
        metrics_test = evaluate(y_test, y_test_pred, "Test (OOS)")

        results.append({
            'model': model_type,
            'val_rmse': metrics_val['rmse'],
            'val_mae': metrics_val['mae'],
            'val_dir_acc': metrics_val['dir_acc'],
            'test_rmse': metrics_test['rmse'],
            'test_mae': metrics_test['mae'],
            'test_dir_acc': metrics_test['dir_acc'],
            'test_ic': metrics_test.get('ic')
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_path, index=False)
    print(f"Benchmark summary written to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    run_benchmark()
