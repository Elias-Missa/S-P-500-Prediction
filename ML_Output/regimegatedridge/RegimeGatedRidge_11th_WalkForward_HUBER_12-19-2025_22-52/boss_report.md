
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
| **SPY Buy & Hold** | 83.51% | 22.97% | 2.73% | 8.42 | 7.04 | 2.03 |
| L/S (Daily Overlap) | 41.86% | 12.64% | 2.53% | 4.99 | 4.97 | 1.07 |
| Long Only (Daily) | 58.36% | 16.95% | 2.41% | 7.02 | 5.56 | 1.50 |
| L/S (Monthly Rebal) | 50.15% | 14.85% | 2.90% | 5.12 | 4.81 | 1.09 |
| Big Move (Daily) | 17.41% | 5.62% | 1.37% | 4.11 | 5.48 | 4.55 |
| Vol Target 15% (Daily) | 104.04% | 27.49% | 5.03% | 5.47 | 5.19 | 1.23 |

### 📉 Drawdown & Trade Stats
| Strategy | Max DD | DD Duration | Win Rate | Profit Factor |
| --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -11.32% | 91 days | 74.2% | 3.19 |
| L/S (Daily Overlap) | -11.86% | 146 days | 62.3% | 2.26 |
| Long Only (Daily) | -11.32% | 91 days | 69.9% | 3.27 |
| L/S (Monthly Rebal) | -13.57% | 130 days | 65.0% | 2.07 |
| Big Move (Daily) | -1.23% | 176 days | 20.7% | 10.16 |
| Vol Target 15% (Daily) | -22.34% | 147 days | 62.3% | 2.32 |
