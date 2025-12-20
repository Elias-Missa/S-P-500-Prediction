
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

> **Note**: Simulation includes 5bps transaction costs per turnover (slippage + comms).

### 📈 Returns & Risk Metrics
| Strategy | Total Return | CAGR | Volatility | Sharpe | Sortino | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | 83.51% | 20.72% | 2.73% | 7.60 | 7.04 | 1.83 |
| L/S (Daily Overlap) | 43.49% | 12.33% | 2.51% | 4.91 | 5.16 | 1.04 |
| L/S (Monthly Rebal) | 50.88% | 14.05% | 2.90% | 4.85 | 4.84 | 1.04 |
| Big Move (Daily) | 17.29% | 5.44% | 1.36% | 4.00 | 5.44 | 4.41 |

### 📉 Drawdown & Trade Stats
| Strategy | Max DD | DD Duration | Win Rate | Profit Factor |
| --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | -11.32% | 91 days | 74.2% | 3.19 |
| L/S (Daily Overlap) | -11.84% | 121 days | 63.2% | 2.36 |
| L/S (Monthly Rebal) | -13.57% | 130 days | 65.5% | 2.10 |
| Big Move (Daily) | -1.23% | 176 days | 22.2% | 10.08 |
