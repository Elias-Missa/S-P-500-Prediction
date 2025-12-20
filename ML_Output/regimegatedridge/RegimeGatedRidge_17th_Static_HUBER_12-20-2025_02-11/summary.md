# ML Run Summary

**Date**: 2025-12-20 02:11:15
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Hyperparameter Tuning
- **Method**: Static (single train/val split)
- **CV Folds**: 1
- **Tuning Data Window**: 2010-01-01 to 2022-12-01
- **Optuna Trials**: 20

## Configuration
- **Model**: `RegimeGatedRidge`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Embargo**: 21 rows (trading days to prevent leakage)
- **Test Start**: 2023-01-01

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
- RMSE: 0.050230
- MAE: 0.040311
- Directional Accuracy: 79.37%
- IC: 0.6780

### Test (Out of Sample)
- RMSE: 0.037847
- MAE: 0.030609
- Directional Accuracy: 61.08%
- IC: 0.1933

#### Always-In Strategy (Sign-Based)
- Total Return: 0.3993
- Sharpe Ratio: 1.04
- Max Drawdown: -0.1025

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0952
- Annualized Return: 0.0339
- Annualized Volatility: 0.0596
- Sharpe Ratio: 0.57
- Max Drawdown: -0.0278
- Trade Count: 4
- Holding Frequency: 11.8%
- Avg Return per Trade: 0.0240

#### Big Move Detection Performance
- Precision (Up): 0.55 (Predicted: 84)
- Recall (Up): 0.16 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 1)
- Recall (Down): 0.00 (Actual: 83)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0092 |
| Std | 0.0360 | 0.0183 |
| Min | -0.1216 | -0.0323 |
| Max | 0.1537 | 0.0772 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 67.7% |
| % Big Up (>3%) | 39.2% | 11.4% |
| % Big Down (<-3%) | 11.2% | 0.1% |

## Features Used
Total Features: 24
List: Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
