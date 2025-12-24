# 💼 BOSS REPORT: Trading Strategy Analysis

## 1. Executive Summary
Analysis of XGBoost model predicting S&P 500 21-day returns.
This report evaluates the model's ability to forecast forward returns and the performance of trading strategies derived from those predictions.

## 2. Model Architecture
Model Type: XGBoost
Mechanism: Machine Learning Regressor.
  - Goal: Predict future returns based on historical patterns.
Structure: Standard Configuration
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

## Stack Decomposition
N/A

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

## Stack Decomposition Analysis
N/A

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
| **SPY Buy & Hold** | 6.29% | 101.09% | 15.85% | 6.38 | 5.12 | 40.86 |
| L/S (Daily Overlap) | 1.42% | 17.53% | 4.81% | 3.65 | inf | 0.00 |
| Long Only (Daily) | 1.42% | 17.53% | 4.81% | 3.65 | inf | 0.00 |
| L/S (Monthly Rebal) | 6.24% | 99.94% | 15.88% | 6.29 | 5.06 | 40.39 |
| Big Move (Daily) | 0.14% | 1.56% | 0.46% | 3.41 | inf | 0.00 |
| Vol Target 15% (Daily) | 1.42% | 17.50% | 4.80% | 3.65 | inf | 0.00 |

### 📉 Drawdown & Trade Stats
| Strategy | Max Drawdown | DD Duration (days) | Win Rate | Profit Factor | Beta vs SPY | Correlation vs SPY | Up Capture | Down Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -2.47% | 4 | 54.5% | 2.14 | 1.05 | 1.000 | 1.00 | 1.00 |
| L/S (Daily Overlap) | 0.00% | 0 | 4.5% | 0.00 | 0.08 | 0.266 | 0.12 | -0.00 |
| Long Only (Daily) | 0.00% | 0 | 4.5% | 0.00 | 0.08 | 0.266 | 0.12 | -0.00 |
| L/S (Monthly Rebal) | -2.47% | 4 | 54.5% | 2.12 | 1.05 | 1.000 | 1.00 | 1.01 |
| Big Move (Daily) | 0.00% | 0 | 4.5% | 0.00 | 0.01 | 0.266 | 0.01 | -0.00 |
| Vol Target 15% (Daily) | 0.00% | 0 | 4.5% | 0.00 | 0.08 | 0.266 | 0.12 | -0.00 |
