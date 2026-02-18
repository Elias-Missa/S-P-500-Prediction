# 💼 BOSS REPORT: Trading Strategy Analysis

## 1. Executive Summary
Static Analysis of Ridge model.
Trained on 10 years history (until 2022), tested on 2023-Present.

## 2. Model Architecture
Model Type: Ridge (Regularized Linear Regression)
Mechanism: Linear approach that balances fit with complexity (L2 Regularization).
  - Pros: Highly robust to noise, less prone to overfitting.
Structure: Alpha=0.1048351095036547

## 3. Validation Methodology
Static Split Validation.
The model is trained ONCE on historical data and tested on a subsequent unseen period.
This tests the model's ability to generalize over time without retraining (Strict OOS).

## 4. Strategy Definitions
A) Long/Short (Daily Overlap): Enters 1/21 position daily. Smooths timing luck.
B) Long Only: Same as L/S but no shorting (Cash instead).
C) Vol Target 15%: Adjusts leverage to target 15% annualized volatility.
D) Big Move Sniper: Only trades when prediction > 3% (Binary).

## 5. Metrics Glossary
Sharpe Ratio: Excess Return / Volatility (>1.0 Good).
Max Drawdown: Deepest peak-to-trough decline.
CAGR: Compound Annual Growth Rate.

## Appendix: Strategy Definitions
To ensure full visibility on how these backtests operate, here are the detailed mechanics of each strategy:

**1. L/S (Daily Overlap)**
- **Concept**: Tells us how the model performs if we trade every single day based on the latest prediction.
- **Mechanics**: Instead of putting 100% of capital into one trade, we enter 1/21th of the position each day. This smooths out volatility and noise. It effectively averages the signals over the last 21 days.
- **Position**: Can be Long (+1), Short (-1), or Neutral. Net position fluctuates smoothly.

**2. Long Only (Daily)**
- **Concept**: Traditional 'Long Only' equity fund logic. No shorting allowed.
- **Mechanics**: If the model predicts UP, we buy. If the model predicts DOWN, we go to Cash (0 position). Uses the same 'Daily Overlap' smoothing as above.
- **Position**: Long (+1) or Cash (0).

**3. L/S (Monthly Rebal)**
- **Concept**: Standard monthly hedge fund rebalancing.
- **Mechanics**: We look at the model's signal only once every 21 days. We ignore all daily fluctuations in between. We enter a trade and hold it 'locked' for the full month.
- **Pros/Cons**: Lower transaction costs, but reacts slower to sudden market changes.

**4. Big Move (Daily)**
- **Concept**: 'Sniper' strategy. Only trade when the model is extremely confident.
- **Mechanics**: We set a high threshold (e.g., 3%). If prediction > 3%, go 100% Long. If prediction < -3%, go 100% Short. Otherwise, sit in Cash.
- **Position**: Binary bets: +1, -1, or 0.

**5. Vol Target 15% (Daily)**
- **Concept**: Risk parity approach. Trade larger when market is calm, smaller when volatile.
- **Mechanics**: We target 15% annualized volatility. Leverage = 15% / Current Vol. Direction follows model.
- **Position**: Variable leverage (Capped at 2x). E.g., if Vol is 30% -> 0.5x position. If Vol is 10% -> 1.5x position.

> **Note**: Simulation includes 5bps transaction costs per turnover (slippage + comms).

## Performance Summary

### 📈 Returns & Risk Metrics
| Strategy | Total Return | CAGR | Volatility | Sharpe Ratio | Sortino Ratio | Calmar Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | 80.20% | 22.54% | 15.26% | 1.48 | 1.36 | 1.20 |
| L/S (Daily Overlap) | 58.23% | 17.16% | 13.68% | 1.26 | 1.25 | 1.35 |
| Long Only (Daily) | 66.11% | 19.15% | 12.58% | 1.52 | 1.31 | 1.49 |
| L/S (Monthly Rebal) | 70.27% | 20.17% | 15.29% | 1.32 | 1.37 | 1.54 |
| Big Move (Daily) | 13.50% | 4.47% | 7.51% | 0.60 | 0.44 | 0.60 |
| Vol Target 15% (Daily) | 137.42% | 34.78% | 27.35% | 1.27 | 1.25 | 1.43 |

### 📉 Drawdown & Trade Stats
| Strategy | Max Drawdown | DD Duration (days) | Win Rate | Profit Factor | Beta vs SPY | Correlation vs SPY | Up Capture | Down Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -18.76% | 90 | 54.9% | 1.31 | 1.00 | 1.000 | 1.00 | 1.00 |
| L/S (Daily Overlap) | -12.71% | 109 | 52.2% | 1.30 | 0.47 | 0.526 | 0.35 | 0.21 |
| Long Only (Daily) | -12.89% | 71 | 45.5% | 1.44 | 0.72 | 0.874 | 0.65 | 0.59 |
| L/S (Monthly Rebal) | -13.10% | 154 | 52.3% | 1.28 | 0.38 | 0.374 | 0.29 | 0.10 |
| Big Move (Daily) | -7.43% | 113 | 23.7% | 1.27 | 0.31 | 0.624 | 0.19 | 0.18 |
| Vol Target 15% (Daily) | -24.30% | 109 | 52.2% | 1.30 | 0.94 | 0.526 | 0.69 | 0.42 |
