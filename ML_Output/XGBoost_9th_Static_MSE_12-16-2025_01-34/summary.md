# ML Run Summary

**Date**: 2025-12-16 01:34:39
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Hyperparameter Tuning
- **Method**: Static (single train/val split)
- **CV Folds**: 1
- **Tuning Data Window**: 2010-01-01 to 2022-12-09
- **Optuna Trials**: 20

## Configuration
- **Model**: `XGBoost`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01

- **Target Scaling Mode**: `vol_scale`

## Training Loss
- **Loss Mode**: mse
- **Prediction Clipping**: None

## Model Description
> **XGBoost (Extreme Gradient Boosting)**: A powerful ensemble method that builds a sequence of decision trees. Each new tree corrects the errors of the previous ones. It is known for high performance and speed.

## Model Parameters
- N Estimators: 100
- Learning Rate: 0.1

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
- RMSE: 0.056359
- MAE: 0.045073
- Directional Accuracy: 67.94%
- IC: 0.6848

### Test (Out of Sample)
- RMSE: 0.044673
- MAE: 0.037368
- Directional Accuracy: 51.25%
- IC: 0.2445

#### Always-In Strategy (Sign-Based)
- Total Return: 4.3822
- Sharpe Ratio: 0.51
- Max Drawdown: -0.9527

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.2701
- Annualized Return: -0.0043
- Annualized Volatility: 0.0571
- Sharpe Ratio: -0.07
- Max Drawdown: -0.7231
- Trade Count: 155
- Holding Frequency: 20.4%
- Avg Return per Trade: -0.0017

#### Big Move Detection Performance
- Precision (Up): 0.76 (Predicted: 25)
- Recall (Up): 0.07 (Actual: 290)
- Precision (Down): 0.08 (Predicted: 130)
- Recall (Down): 0.13 (Actual: 83)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0168 | -0.0066 |
| Std | 0.0357 | 0.0235 |
| Min | -0.1216 | -0.0737 |
| Max | 0.1537 | 0.0606 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 73.6% | 42.7% |
| % Big Up (>3%) | 38.1% | 3.3% |
| % Big Down (<-3%) | 10.9% | 17.1% |

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
