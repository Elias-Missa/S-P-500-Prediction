# ML Run Summary

**Date**: 2025-12-04 13:03:09

## Configuration
- **Model**: `MLP`
- **Target Horizon**: 21 days
- **Train Window**: 5 years
- **Val Window**: 6 months
- **Buffer**: 30 days
- **Test Start**: 2023-01-01

## Model Parameters
- Hidden Layers: (64, 32)

## Metrics
### Validation
- RMSE: 0.160281
- MAE: 0.090428
- Directional Accuracy: 61.90%

### Test (Out of Sample)
- RMSE: 0.054022
- MAE: 0.039964
- Directional Accuracy: 57.16%

## Features Used
Total Features: 34
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap, rsi_mrktw_14, dd_mrktw_21, mm_mrktw_21, dd_mrktw_127, mm_mrktw_127, dd_mrktw_255, mm_mrktw_255, rsi_spx_14, dd_spx_255, net_1W, net_1M, net_3M, net_1Y, spx.100.Bb.allnewB.bottomPrec30SigMatrix_60_30, spx.100.Bb.allnewB.bottomPrec60SigMatrix_60_30, spx.100.Tb.allnewB.topPrec30SigMatrix_60_30, spx.100.Tb.allnewB.topPrec60SigMatrix_60_30
