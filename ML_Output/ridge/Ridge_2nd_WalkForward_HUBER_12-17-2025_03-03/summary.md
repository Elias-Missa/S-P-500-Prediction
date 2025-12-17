# ML Run Summary

**Date**: 2025-12-17 03:03:29
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `Ridge`
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
- RMSE: 0.038220
- MAE: 0.031722
- Directional Accuracy: 67.51%
- IC: 0.3717

### Test (Out of Sample)
- RMSE: 0.037704
- MAE: 0.029942
- Directional Accuracy: 66.49%
- IC: 0.1880

#### Always-In Strategy (Sign-Based)
- Total Return: 7793.1882
- Sharpe Ratio: 5.41
- Max Drawdown: -0.9264

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 4.9620
- Annualized Return: 0.6692
- Annualized Volatility: 0.3528
- Sharpe Ratio: 1.90
- Max Drawdown: -0.6743
- Trade Count: 88
- Holding Frequency: 11.9%
- Avg Return per Trade: 0.0223

#### Big Move Detection Performance
- Precision (Up): 0.48 (Predicted: 88)
- Recall (Up): 0.14 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0213
- T-statistic: +2.64
- P-value: 0.0092
- Monotonicity: +0.733
- Top Decile Mean: +0.0242
- Bottom Decile Mean: +0.0029

**Coverage vs Performance:**
- Best Threshold: 0.0060
- Coverage at Best: 75.0%
- Sharpe at Best: 5.50
- Coverage-Sharpe Corr: +0.952

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0029 |
| Q2 | +0.0072 |
| Q3 | +0.0144 |
| Q4 | +0.0218 |
| Q5 | +0.0161 |
| Q6 | +0.0243 |
| Q7 | +0.0160 |
| Q8 | +0.0240 |
| Q9 | +0.0217 |
| Q10 | +0.0242 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
