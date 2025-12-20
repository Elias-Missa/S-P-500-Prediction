
## 💼 Boss Report: Trading Strategy Analysis
### 📘 Strategy Definitions
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

**5. Vol Target 15% (Daily)**
- **Concept**: Risk parity approach. Trade larger when market is calm, smaller when volatile.
- **Mechanics**: We target 15% annualized volatility. Leverage = 15% / Current Vol. Direction follows model.
- **Position**: Variable leverage (Capped at 2x). E.g., if Vol is 30% -> 0.5x position. If Vol is 10% -> 1.5x position.

> **Note**: Simulation includes 5bps transaction costs per turnover (slippage + comms).

### 📈 Returns & Risk Metrics
| Strategy | Total Return | CAGR | Volatility | Sharpe | Sortino | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | 80.20% | 22.21% | 15.16% | 1.46 | 1.34 | 1.18 |
| L/S (Daily Overlap) | 44.45% | 13.34% | 13.49% | 0.99 | 1.02 | 0.71 |
| Long Only (Daily) | 57.23% | 16.66% | 13.26% | 1.26 | 1.17 | 0.89 |
| L/S (Monthly Rebal) | 41.86% | 12.65% | 15.20% | 0.83 | 0.87 | 0.58 |
| Big Move (Daily) | 9.47% | 3.13% | 8.52% | 0.37 | 0.22 | 0.28 |
| Vol Target 15% (Daily) | 42.93% | 12.93% | 14.10% | 0.92 | 0.97 | 0.69 |

### 📉 Drawdown & Trade Stats
| Strategy | Max DD | DD Duration | Win Rate | Profit Factor |
| --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -18.76% | 92 days | 54.2% | 1.31 |
| L/S (Daily Overlap) | -18.76% | 106 days | 50.3% | 1.24 |
| Long Only (Daily) | -18.76% | 92 days | 50.0% | 1.32 |
| L/S (Monthly Rebal) | -21.92% | 144 days | 50.9% | 1.18 |
| Big Move (Daily) | -11.14% | 177 days | 14.7% | 1.28 |
| Vol Target 15% (Daily) | -18.64% | 137 days | 50.3% | 1.20 |
