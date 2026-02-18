# 💼 BOSS REPORT: Trading Strategy Analysis

## 1. Executive Summary
Static Analysis of Ridge model.
Trained on 10 years history (until 2022), tested on 2023-Present.

## 2. Model Architecture
Model Type: Ridge (Static)
Structure: Alpha=0.10351493687874229

## 3. Validation Methodology
Static Split (Single Train/Test). Tests model robustness over long horizons without retraining.

## 4. Strategy Definitions
Standard Long/Short strategies simulated on test data.

## 5. Metrics Glossary
Sharpe (>1 Good), Max Drawdown (Pain), CAGR (Growth).

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
| **SPY Buy & Hold** | 0.00% | 0.00% | 0.00% | 0.00 | inf | 0.00 |
| L/S (Daily Overlap) | -0.83% | -0.29% | 0.03% | -8.85 | -3.78 | -0.35 |
| Long Only (Daily) | -0.43% | -0.15% | 0.02% | -7.39 | -3.28 | -0.35 |
| L/S (Monthly Rebal) | -0.95% | -0.33% | 0.18% | -1.84 | -0.21 | -0.35 |
| Big Move (Daily) | -0.45% | -0.16% | 0.02% | -9.44 | -4.15 | -0.35 |
| Vol Target 15% (Daily) | -1.65% | -0.57% | 0.06% | -8.84 | -3.78 | -0.35 |

### 📉 Drawdown & Trade Stats
| Strategy | Max Drawdown | DD Duration (days) | Win Rate | Profit Factor | Beta vs SPY | Correlation vs SPY | Up Capture | Down Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | 0.00% | 0 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| L/S (Daily Overlap) | -0.83% | 709 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| Long Only (Daily) | -0.43% | 709 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| L/S (Monthly Rebal) | -0.95% | 729 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| Big Move (Daily) | -0.45% | 662 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
| Vol Target 15% (Daily) | -1.65% | 709 | 0.0% | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 |
