# Summary of Recent Changes

## Data ingestion and quality controls
- Added FRED fallbacks for PMI and consumer sentiment while logging which source ticker succeeded, ensuring macro inputs are not silently missing.
- Recorded quality control indicators for missing CPC and NYA50 series, tracking PMI availability, and flagging when NYA50 is proxied from SPY price action.
- Exported a daily data quality report with missing/zero rates and min/max values to `Output/data_quality_report.csv` for transparency in downstream analysis.

## Feature engineering pipeline
- Propagated loader quality flags into the engineered feature set so downstream models can condition on data health.
- Added University of Michigan consumer sentiment z-scores as a macro feature alongside PMI and yield-curve inputs.
- Attached raw SPY prices plus forward-return and log-return targets, then sliced to 2010 and dropped incomplete tail rows to produce `Output/final_features_with_target.csv` for modeling.

## Model training and tuning
- Preserved configuration defaults by resolving tuned parameters into per-run dictionaries rather than mutating global settings during Optuna sweeps.
- Standardized deep-learning hyperparameter resolution for LSTM/CNN models and trained them with sequence-aware loaders without leaking target shifts.
- Reported expanded validation/test metrics (RMSE, MAE, directional accuracy, IC, tail stats) and logged plots/summary artifacts via the experiment logger.

## Benchmarking
- Added a benchmark runner that trains a configurable suite of baseline models on a shared split, optionally tunes with Optuna, and writes a consolidated `ML_Output/benchmark_summary.csv` for quick comparison.
