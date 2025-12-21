# 💼 BOSS REPORT: Trading Strategy Analysis

## 1. Executive Summary
Analysis of TFT model predicting S&P 500 21-day returns.
This report evaluates the model's ability to forecast forward returns and the performance of trading strategies derived from those predictions.

## 2. Model Architecture
Model Type: TFT (Temporal Fusion Transformer)
Mechanism: Advanced Deep Learning architecture for time-series forecasting.
  - Gating Mechanisms: Learns to suppress unnecessary features (Variable Selection Network).
  - LSTM Layers: Captures local time dependencies and sequence patterns.
  - Temporal Attention: Identifies long-range patterns and crucial past events.
  - Interpretability: Provides built-in feature importance and attention weights.
Structure: Hidden=64, Heads=4, Layers=2 (Default)
Input Features: 24 features selected for predictive power, including technicals, macro indicators, and sentiment.

## 3. Validation Methodology
Technique: Strict Walk-Forward Validation (No Look-Ahead).
Why: To simulate real-world trading exactly as it would have happened.
Process:
  1. Rolling Train Window: The model is trained on the past 10 years of data.
  2. Safety Embargo: A 21-day gap is enforced between training and testing to prevent 'target leakage' (ensure we don't trade on data that wasn't public yet).
  3. Out-of-Sample Test: The model predicts the NEXT month of unseen data.
  4. Step Forward: The window rolls forward 1 month, and the process repeats (34 times).
Result: A composite track record of predictions made only on 'future' data.

## 4. Strategy Definitions
A) Long/Short (Daily Overlap): 
   - Concept: Enters a small portion (1/21) of the position every day based on the latest signal.
   - Effect: Smooths out timing luck. Effectively holds the average opinion of the model over the last month.
   - Result: Lower volatility, higher Sharpe potential due to time-diversification.

B) Long Only: 
   - Concept: Same smoothing as above, but never shorts. If the model is bearish, it goes to Cash (0).

C) Vol Target 15%: 
   - Concept: Institutional-style risk parity. 
   - Logic: When market vol is low (10%), we lever up (1.5x) to target 15%. When market vol is high (30%), we size down (0.5x).
   - Goal: Stable returns regardless of market environment.

D) Big Move Sniper: 
   - Concept: Only trades when the model predicts a massive move (>3%). Binary bet (All-in or Cash).

## 5. Metrics Glossary
Sharpe Ratio: Excess Return / Volatility. Measures return per unit of risk. >1.0 is Good, >2.0 is Exceptional.
Sortino Ratio: Like Sharpe, but ignores 'good' volatility (upside). Only penalizes downside risk.
CAGR: Compound Annual Growth Rate (Geometric). The actual annual growth rate of your capital.
Max Drawdown: Deepest peak-to-trough decline. A measure of 'pain'.

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
| **SPY Buy & Hold** | 80.20% | 22.21% | 15.16% | 1.46 | 1.34 | 1.18 |
| L/S (Daily Overlap) | 40.23% | 12.20% | 13.02% | 0.94 | 0.93 | 0.83 |
| Long Only (Daily) | 54.59% | 15.99% | 13.56% | 1.18 | 1.10 | 0.96 |
| L/S (Monthly Rebal) | 36.96% | 11.30% | 15.19% | 0.74 | 0.75 | 0.60 |
| Big Move (Daily) | 26.71% | 8.40% | 5.11% | 1.64 | 1.45 | 2.40 |
| Vol Target 15% (Daily) | 43.76% | 13.16% | 14.47% | 0.91 | 0.90 | 0.96 |

### 📉 Drawdown & Trade Stats
| Strategy | Max Drawdown | DD Duration (days) | Win Rate | Profit Factor | Beta vs SPY | Correlation vs SPY | Up Capture | Down Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -18.76% | 92 | 54.2% | 1.31 | 1.00 | 1.000 | 1.00 | 1.00 |
| L/S (Daily Overlap) | -14.67% | 120 | 51.5% | 1.23 | 0.74 | 0.866 | 0.64 | 0.66 |
| Long Only (Daily) | -16.64% | 94 | 51.8% | 1.28 | 0.86 | 0.960 | 0.80 | 0.82 |
| L/S (Monthly Rebal) | -18.76% | 336 | 52.0% | 1.16 | 0.84 | 0.837 | 0.73 | 0.79 |
| Big Move (Daily) | -3.49% | 168 | 32.8% | 1.55 | 0.24 | 0.720 | 0.24 | 0.20 |
| Vol Target 15% (Daily) | -13.74% | 165 | 51.5% | 1.20 | 0.79 | 0.823 | 0.77 | 0.81 |
