import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def calculate_max_drawdown(equity_curve):
    """Calculates Maximum Drawdown and Duration."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    # Duration (approximate days underwater)
    is_underwater = drawdown < 0
    if is_underwater.sum() == 0:
        return max_dd, 0
    duration_series = is_underwater.astype(int).groupby((~is_underwater).cumsum()).cumsum()
    max_duration = duration_series.max()
    
    return max_dd, max_duration

def calculate_sortino_ratio(returns, target_return=0.0, periods=252):
    """Sortino: Penalizes only downside volatility."""
    mean_ret = returns.mean() * periods
    downside = returns[returns < target_return]
    if len(downside) == 0: return np.inf
    
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(periods)
    if downside_std == 0: return 0.0
    
    return (mean_ret - target_return) / downside_std

def calculate_calmar_ratio(ann_return, max_dd):
    """Calmar: Annual Return / Max Drawdown."""
    if max_dd == 0: return 0.0
    return ann_return / abs(max_dd)

def generate_boss_metrics(strategy_returns, periods=252):
    """Generates the dictionary of metrics for the report."""
    # 1. Basics
    total_ret = np.prod(1 + strategy_returns) - 1
    # Geometric CAGR: (1 + total_ret)^(periods/N) - 1
    n_days = len(strategy_returns)
    if n_days > 0:
        ann_ret = (1 + total_ret)**(periods / n_days) - 1
    else:
        ann_ret = 0.0
    ann_vol = strategy_returns.std() * np.sqrt(periods)
    
    # 2. Risk
    equity = (1 + strategy_returns).cumprod()
    max_dd, max_dur = calculate_max_drawdown(equity)
    
    # 3. Ratios
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    sortino = calculate_sortino_ratio(strategy_returns, periods=periods)
    calmar = calculate_calmar_ratio(ann_ret, max_dd)
    
    # 4. Win/Loss
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0
    
    return {
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{ann_ret:.2%}",
        "Volatility": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Max DD": f"{max_dd:.2%}",
        "DD Duration": f"{max_dur} days",
        "Win Rate": f"{win_rate:.1%}",
        "Profit Factor": f"{profit_factor:.2f}"
    }

def generate_boss_metrics_enhanced(strategy_returns, market_returns, periods=252):
    """Generates enhanced dictionary of metrics for the boss report."""
    # Basic metrics (from original function)
    total_ret = np.prod(1 + strategy_returns) - 1
    # Geometric CAGR
    n_days = len(strategy_returns)
    if n_days > 0:
        ann_ret = (1 + total_ret)**(periods / n_days) - 1
    else:
        ann_ret = 0.0
    ann_vol = strategy_returns.std() * np.sqrt(periods)
    
    # Risk metrics
    equity = (1 + strategy_returns).cumprod()
    max_dd, max_dur = calculate_max_drawdown(equity)
    
    # Ratios
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    sortino = calculate_sortino_ratio(strategy_returns, periods=periods)
    calmar = calculate_calmar_ratio(ann_ret, max_dd)
    
    # Win/Loss
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0
    
    # Additional metrics
    # Beta vs Market (if aligned)
    if len(strategy_returns) == len(market_returns):
        # Sanitize inputs for correlation/beta
        strat_clean = np.nan_to_num(strategy_returns, nan=0.0, posinf=0.0, neginf=0.0)
        market_clean = np.nan_to_num(market_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        covariance = np.cov(strat_clean, market_clean)[0, 1]
        market_variance = np.var(market_clean)
        strat_variance = np.var(strat_clean)
        
        # Check variance thresholds to avoid RuntimeWarning (div by zero)
        beta = covariance / market_variance if market_variance > 1e-8 else 0
        
        if market_variance > 1e-8 and strat_variance > 1e-8:
            correlation = np.corrcoef(strat_clean, market_clean)[0, 1]
        else:
            correlation = 0
    
    # Skewness and Kurtosis
    ret_skew = skew(strategy_returns)
    ret_kurt = kurtosis(strategy_returns)
    
    # Value at Risk (VaR) - 5th percentile
    var_95 = np.percentile(strategy_returns, 5)
    
    # Conditional VaR (CVaR) - Expected loss given VaR breach
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean() if (strategy_returns <= var_95).sum() > 0 else 0
    
    # Best/Worst month (approximate using rolling 21-day)
    rets_series = pd.Series(strategy_returns)
    rolling_21d = rets_series.rolling(21).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    best_month = rolling_21d.max() if len(rolling_21d.dropna()) > 0 else 0
    worst_month = rolling_21d.min() if len(rolling_21d.dropna()) > 0 else 0
    
    # Number of trades (position changes)
    position_changes = (strategy_returns != 0).sum()
    
    # Average trade return
    avg_trade_ret = strategy_returns[strategy_returns != 0].mean() if (strategy_returns != 0).sum() > 0 else 0
    
    # Capture Ratios (Up/Down)
    up_capture, down_capture = calculate_capture_ratios(strategy_returns, market_returns)
    
    return {
        # Core metrics
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{ann_ret:.2%}",
        "Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "DD Duration (days)": f"{max_dur}",
        "Win Rate": f"{win_rate:.1%}",
        "Profit Factor": f"{profit_factor:.2f}",
        
        # Enhanced metrics
        "Beta vs SPY": f"{beta:.2f}",
        "Correlation vs SPY": f"{correlation:.3f}",
        "Up Capture": f"{up_capture:.2f}",
        "Down Capture": f"{down_capture:.2f}",
        "Best Month": f"{best_month:.2%}",
        "Worst Month": f"{worst_month:.2%}",
        "Skewness": f"{ret_skew:.2f}",
        "Kurtosis": f"{ret_kurt:.2f}",
        "VaR (95%)": f"{var_95:.2%}",
        "CVaR (95%)": f"{cvar_95:.2%}",
        "Avg Trade Return": f"{avg_trade_ret:.2%}",
        "Number of Trades": f"{position_changes}",
    }

def calculate_capture_ratios(strategy_returns, benchmark_returns):
    """Calculates Up and Down Capture Ratios."""
    # Ensure alignment
    if len(strategy_returns) != len(benchmark_returns):
        return 0.0, 0.0
        
    # Sanitize
    strat = np.nan_to_num(strategy_returns, nan=0.0)
    bench = np.nan_to_num(benchmark_returns, nan=0.0)
    
    # Up Capture: Avg Strat Return when Bench > 0 / Avg Bench Return when Bench > 0
    up_mask = bench > 0
    if up_mask.sum() > 0:
        up_capture = strat[up_mask].mean() / bench[up_mask].mean()
    else:
        up_capture = 0.0
        
    # Down Capture: Avg Strat Return when Bench < 0 / Avg Bench Return when Bench < 0
    down_mask = bench < 0
    if down_mask.sum() > 0:
        down_capture = strat[down_mask].mean() / bench[down_mask].mean()
    else:
        down_capture = 0.0
        
    return up_capture, down_capture