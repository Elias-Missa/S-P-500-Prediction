# ML Run Summary

**Date**: 2025-12-17 02:47:33
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `Transformer`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Embargo**: 1 rows (trading days to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: False
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: huber
- **Huber Delta**: 1.0
- **Prediction Clipping**: None

## Model Description

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
- RMSE: 0.040313
- MAE: 0.035631
- Directional Accuracy: 65.91%
- IC: 0.2182

### Test (Out of Sample)
- RMSE: 0.033354
- MAE: 0.027186
- Directional Accuracy: 45.45%
- IC: -0.1909

#### Always-In Strategy (Sign-Based)
- Total Return: -0.0115
- Sharpe Ratio: -0.12
- Max Drawdown: -0.0761

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.0145
- Annualized Return: -0.0147
- Annualized Volatility: 0.0504
- Sharpe Ratio: -0.29
- Max Drawdown: -0.0403
- Trade Count: 2
- Holding Frequency: 18.2%
- Avg Return per Trade: -0.0067

#### Big Move Detection Performance
- Precision (Up): 0.00 (Predicted: 2)
- Recall (Up): 0.00 (Actual: 1)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 1)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0000
- T-statistic: +0.00
- P-value: 1.0000
- Monotonicity: +0.000
- Top Decile Mean: +0.0000
- Bottom Decile Mean: +0.0000

**Coverage vs Performance:**
- Best Threshold: 0.0375
- Coverage at Best: 9.1%
- Sharpe at Best: 1.10
- Coverage-Sharpe Corr: -0.222

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
