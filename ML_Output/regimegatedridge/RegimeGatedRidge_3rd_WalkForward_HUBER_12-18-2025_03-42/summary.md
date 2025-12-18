# ML Run Summary

**Date**: 2025-12-18 03:42:08
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `RegimeGatedRidge`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Embargo**: 21 rows (trading days to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: False
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: huber
- **Huber Delta**: 1.0
- **Prediction Clipping**: None

## Model Description
> **Regime-Gated Ridge**: A specialized model that adapts to market volatility. It splits the training data into 'Low Volatility' and 'High Volatility' regimes based on the `RV_Ratio` (Realized Volatility Ratio). Two separate Ridge Regression models are trained—one for each regime. During prediction, the model detects the current regime and routes the input to the appropriate sub-model. This allows it to be aggressive in calm markets and conservative (or different) in turbulent ones.

## Model Parameters

## Metrics Explanation
### Standard Metrics
- **RMSE (Root Mean Squared Error)**: The average magnitude of the prediction error. Lower is better.
- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual returns. Lower is better.
- **Directional Accuracy**: The percentage of time the model correctly predicted the *sign* (Up/Down) of the return. >50% is the goal.
### Advanced Metrics
- **IC (Information Coefficient)**: The Spearman correlation between predictions and actuals. Measures how well the model ranks returns. >0.05 is good, >0.10 is excellent.
- **Strategy Return**: The cumulative return of a simple strategy: Long if Pred > 0, Short if Pred < 0.
- **Sharpe Ratio**: Annualized risk-adjusted return of the strategy. >1.0 is good.
- **Max Drawdown**: The largest percentage drop from a peak in the strategy's equity curve. Smaller magnitude (closer to 0) is better.
### Big Shift Analysis (>5%)
Focuses on extreme moves (market crashes or rallies) greater than 5% in a month.
- **Precision**: When the model predicts a Big Move, how often is it right? (High Precision = Few False Alarms).
- **Recall**: When a Big Move actually happens, how often did the model predict it? (High Recall = Few Missed Opportunities).

## Results
### Validation (In-Sample / Tuning)
- RMSE: 0.038147
- MAE: 0.031746
- Directional Accuracy: 67.67%
- IC: 0.3734

### Test (Out of Sample)
- RMSE: 0.037269
- MAE: 0.029844
- Directional Accuracy: 65.27%
- IC: 0.2035

#### Always-In Strategy (Sign-Based)
- Total Return: 0.5194
- Sharpe Ratio: 1.31
- Max Drawdown: -0.0992

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0378
- Annualized Return: 0.0143
- Annualized Volatility: 0.0505
- Sharpe Ratio: 0.28
- Max Drawdown: -0.0278
- Trade Count: 3
- Holding Frequency: 8.8%
- Avg Return per Trade: 0.0135

#### Big Move Detection Performance
- Precision (Up): 0.47 (Predicted: 79)
- Recall (Up): 0.13 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0218
- T-statistic: +2.79
- P-value: 0.0060
- Monotonicity: +0.915
- Top Decile Mean: +0.0250
- Bottom Decile Mean: +0.0033

**Coverage vs Performance:**
- Best Threshold: 0.0000
- Coverage at Best: 100.0%
- Sharpe at Best: 1.31
- Coverage-Sharpe Corr: +0.878

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0033 |
| Q2 | +0.0085 |
| Q3 | +0.0163 |
| Q4 | +0.0147 |
| Q5 | +0.0191 |
| Q6 | +0.0166 |
| Q7 | +0.0182 |
| Q8 | +0.0285 |
| Q9 | +0.0225 |
| Q10 | +0.0250 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0144
- **Threshold Std**: 0.0164
- **Threshold Range**: [0.0026, 0.0540]
- **Val Sharpe (avg)**: 2.44
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 19
- **Per-Fold τ**: [0.0050, 0.0062, 0.0075, 0.0099, 0.0112, 0.0092, 0.0097, 0.0037, 0.0050, 0.0050, 0.0050, 0.0524, 0.0526, 0.0540, 0.0529, 0.0530, 0.0049, 0.0050, 0.0050, 0.0044, 0.0062, 0.0080, 0.0087, 0.0072, 0.0028, 0.0026, 0.0033, 0.0134, 0.0139, 0.0143, 0.0142, 0.0150, 0.0146, 0.0030]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0101 |
| Std | 0.0360 | 0.0170 |
| Min | -0.1216 | -0.0292 |
| Max | 0.1537 | 0.0656 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 72.7% |
| % Big Up (>3%) | 39.2% | 10.7% |
| % Big Down (<-3%) | 11.2% | 0.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.4208, std=0.2959

**Best 3 folds by IC**:
- Fold 2: IC=0.8864, Dir Acc=100.0%, Test: 2023-03-01 to 2023-03-31
- Fold 32: IC=0.8758, Dir Acc=54.5%, Test: 2025-09-01 to 2025-09-30
- Fold 24: IC=0.8668, Dir Acc=73.9%, Test: 2025-01-01 to 2025-01-31

**Worst 3 folds by IC**:
- Fold 17: IC=-0.1208, Dir Acc=23.8%, Test: 2024-06-01 to 2024-06-28
- Fold 21: IC=-0.1621, Dir Acc=91.3%, Test: 2024-10-01 to 2024-10-31
- Fold 7: IC=-0.1759, Dir Acc=52.2%, Test: 2023-08-01 to 2023-08-31

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
