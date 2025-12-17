"""
Metrics Module

Comprehensive strategy evaluation with multiple trading policies and volatility targeting.

Supported Policies:
- long_flat: Long if pred > 0, flat otherwise
- long_short: Long if pred > 0, short if pred < 0
- thresholded: Long if pred > threshold, short if pred < -threshold, flat otherwise
- continuous_sizing: Position size proportional to prediction magnitude (scaled)

Optional Overlays:
- Volatility targeting: Scale positions toward target vol using realized vol
- Transaction costs: (Disabled by default, behind a flag)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from . import config


# ============================================================================
# Configuration & Constants
# ============================================================================

# Available trading policies
TRADING_POLICIES = ['long_flat', 'long_short', 'thresholded', 'continuous_sizing']

# Default volatility targeting parameters
DEFAULT_TARGET_VOL = 0.15  # 15% annualized target volatility
DEFAULT_VOL_LOOKBACK = 21  # 21-day rolling vol window
DEFAULT_MAX_LEVERAGE = 2.0  # Maximum leverage allowed
DEFAULT_MIN_LEVERAGE = 0.5  # Minimum leverage (position scaling floor)


def get_annualization_factor(frequency=None):
    """
    Get the appropriate annualization factor based on data frequency.
    
    Args:
        frequency: "daily" or "monthly" (defaults to config.DATA_FREQUENCY)
        
    Returns:
        int: 252 for daily, 12 for monthly
    """
    freq = frequency or getattr(config, 'DATA_FREQUENCY', 'daily')
    return 252 if freq == "daily" else 12


# ============================================================================
# Position Sizing Functions
# ============================================================================

def position_from_pred(pred, policy='long_short', threshold=0.03, scale_factor=1.0):
    """
    Convert predictions to position sizes based on trading policy.
    
    Args:
        pred: Array of predictions (expected returns)
        policy: Trading policy - one of:
            - 'long_flat': Long if pred > 0, flat otherwise (position in [0, 1])
            - 'long_short': Long if pred > 0, short if pred < 0 (position in [-1, 1])
            - 'thresholded': Long/short only if |pred| > threshold (position in [-1, 0, 1])
            - 'continuous_sizing': Position proportional to prediction (scaled)
        threshold: Threshold for 'thresholded' policy
        scale_factor: Scaling factor for 'continuous_sizing' (pred / scale_factor = position)
        
    Returns:
        Array of position sizes (typically in [-1, 1] range before leverage)
    """
    pred = np.array(pred)
    
    if policy == 'long_flat':
        # Long if positive prediction, flat otherwise
        positions = np.where(pred > 0, 1.0, 0.0)
        
    elif policy == 'long_short':
        # Long if positive, short if negative
        positions = np.sign(pred)
        
    elif policy == 'thresholded':
        # Only trade when prediction exceeds threshold
        positions = np.zeros_like(pred)
        positions[pred > threshold] = 1.0
        positions[pred < -threshold] = -1.0
        
    elif policy == 'continuous_sizing':
        # Position size proportional to prediction magnitude
        # Normalize by scale_factor (e.g., std of predictions)
        if scale_factor <= 0:
            scale_factor = max(np.std(pred), 1e-6)
        positions = np.clip(pred / scale_factor, -1.0, 1.0)
        
    else:
        raise ValueError(f"Unknown policy: {policy}. Choose from {TRADING_POLICIES}")
    
    return positions


def apply_volatility_targeting(positions, returns_for_vol, 
                               target_vol=DEFAULT_TARGET_VOL,
                               vol_lookback=DEFAULT_VOL_LOOKBACK,
                               max_leverage=DEFAULT_MAX_LEVERAGE,
                               min_leverage=DEFAULT_MIN_LEVERAGE,
                               frequency=None):
    """
    Apply volatility targeting overlay to position sizes.
    
    Scales positions to target a specific annualized volatility using
    realized (historical) volatility.
    
    Args:
        positions: Array of raw position sizes
        returns_for_vol: Returns series to compute realized vol (e.g., daily SPY returns)
        target_vol: Target annualized volatility (default 15%)
        vol_lookback: Rolling window for realized vol calculation
        max_leverage: Maximum position scaling allowed
        min_leverage: Minimum position scaling allowed
        frequency: "daily" or "monthly" for annualization
        
    Returns:
        Array of vol-scaled positions
    """
    positions = np.array(positions)
    returns_for_vol = np.array(returns_for_vol)
    
    # Get annualization factor
    ann_factor = get_annualization_factor(frequency)
    
    # Ensure returns_for_vol has same length as positions
    if len(returns_for_vol) != len(positions):
        raise ValueError(
            f"returns_for_vol length ({len(returns_for_vol)}) must match positions length ({len(positions)})"
        )
    
    # Compute rolling realized volatility (annualized)
    # Use pandas for rolling calculation
    returns_series = pd.Series(returns_for_vol)
    rolling_vol = returns_series.rolling(window=vol_lookback, min_periods=max(1, vol_lookback // 2)).std()
    rolling_vol_ann = rolling_vol * np.sqrt(ann_factor)
    
    # Fill NaN with target vol (for warmup period)
    rolling_vol_ann = rolling_vol_ann.fillna(target_vol)
    
    # Compute vol scalar: target_vol / realized_vol
    # Shift by 1 to avoid look-ahead bias (use yesterday's vol estimate)
    vol_scalar = (target_vol / rolling_vol_ann).shift(1).fillna(1.0)
    
    # Clip leverage to bounds
    vol_scalar = vol_scalar.clip(lower=min_leverage, upper=max_leverage)
    
    # Apply vol scaling to positions
    scaled_positions = positions * vol_scalar.values
    
    return scaled_positions, vol_scalar.values


# ============================================================================
# Core Metrics Functions
# ============================================================================

def tail_weighted_mse(y_pred, y_true, threshold=0.03, alpha=4.0):
    """
    Tail-weighted MSE loss for PyTorch models.
    
    Applies higher weight to samples where the actual return exceeds
    the threshold, making the model focus more on big moves.
    
    Args:
        y_pred: Predicted values tensor of shape (batch_size, 1) or (batch_size,)
        y_true: Actual values tensor of shape (batch_size, 1) or (batch_size,)
        threshold: Absolute return threshold that defines a 'big' move
        alpha: Additional weight multiplier applied to big moves
               (total weight for big moves = 1 + alpha)
    
    Returns:
        Scalar tensor with the weighted MSE loss
    """
    diff = y_pred - y_true
    weights = 1.0 + alpha * (y_true.abs() > threshold).float()
    return (weights * diff.pow(2)).mean()


def calculate_ic(y_true, y_pred):
    """
    Calculates the Information Coefficient (Spearman Correlation).
    """
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def calculate_hit_rate(y_true, y_pred):
    """
    Calculates directional accuracy (hit rate).
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Float: Fraction of times sign(pred) == sign(actual)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Only count periods where actual != 0
    nonzero_mask = y_true != 0
    if np.sum(nonzero_mask) == 0:
        return 0.0
    
    correct = np.sign(y_pred[nonzero_mask]) == np.sign(y_true[nonzero_mask])
    return np.mean(correct)


def calculate_decile_spread(y_true, y_pred):
    """
    Calculates the decile spread: difference in mean actual return
    between top decile and bottom decile of predictions.
    
    This measures whether high predictions correspond to high returns.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Float: Mean return (top decile) - Mean return (bottom decile)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_pred) < 10:
        return 0.0
    
    # Compute percentiles
    top_threshold = np.percentile(y_pred, 90)
    bottom_threshold = np.percentile(y_pred, 10)
    
    top_mask = y_pred >= top_threshold
    bottom_mask = y_pred <= bottom_threshold
    
    if np.sum(top_mask) == 0 or np.sum(bottom_mask) == 0:
        return 0.0
    
    mean_top = np.mean(y_true[top_mask])
    mean_bottom = np.mean(y_true[bottom_mask])
    
    return mean_top - mean_bottom


# ============================================================================
# Strategy Evaluation Functions
# ============================================================================

def compute_strategy_returns(positions, actual_returns, transaction_cost=0.0, 
                             apply_tc=False):
    """
    Compute strategy returns given positions and actual returns.
    
    Args:
        positions: Array of position sizes
        actual_returns: Array of actual period returns
        transaction_cost: Cost per unit of turnover (e.g., 0.001 = 10 bps)
        apply_tc: Whether to apply transaction costs
        
    Returns:
        Array of strategy returns
    """
    positions = np.array(positions)
    actual_returns = np.array(actual_returns)
    
    # Strategy returns = position * actual return
    strategy_returns = positions * actual_returns
    
    # Apply transaction costs if enabled
    if apply_tc and transaction_cost > 0:
        # Compute turnover (absolute change in position)
        turnover = np.abs(np.diff(positions, prepend=0))
        # Subtract transaction costs
        strategy_returns = strategy_returns - (turnover * transaction_cost)
    
    return strategy_returns


def compute_equity_curve(returns):
    """
    Compute compounded equity curve from returns.
    
    Args:
        returns: Array of period returns
        
    Returns:
        Array of equity values (starts at 1.0)
    """
    returns = np.array(returns)
    return np.cumprod(1 + returns)


def compute_max_drawdown(equity_curve):
    """
    Compute maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Float: Maximum drawdown (negative value)
    """
    equity_curve = np.array(equity_curve)
    if len(equity_curve) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)


def evaluate_policy(y_true, y_pred, policy='long_short', threshold=0.03,
                    returns_for_vol=None, apply_vol_targeting=False,
                    target_vol=DEFAULT_TARGET_VOL, vol_lookback=DEFAULT_VOL_LOOKBACK,
                    max_leverage=DEFAULT_MAX_LEVERAGE, min_leverage=DEFAULT_MIN_LEVERAGE,
                    transaction_cost=0.0, apply_tc=False,
                    frequency=None, scale_factor=None):
    """
    Evaluate a trading policy on predictions vs actuals.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        policy: Trading policy (see position_from_pred)
        threshold: Threshold for thresholded policy
        returns_for_vol: Returns for vol calculation (required if apply_vol_targeting=True)
        apply_vol_targeting: Whether to apply volatility targeting overlay
        target_vol: Target volatility for vol targeting
        vol_lookback: Lookback window for realized vol
        max_leverage: Max leverage for vol targeting
        min_leverage: Min leverage for vol targeting
        transaction_cost: Cost per unit turnover
        apply_tc: Whether to apply transaction costs
        frequency: "daily" or "monthly" for annualization
        scale_factor: Scale factor for continuous_sizing policy
        
    Returns:
        Dictionary with comprehensive strategy metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get annualization factor
    ann_factor = get_annualization_factor(frequency)
    
    # 1. Compute raw positions from predictions
    if scale_factor is None and policy == 'continuous_sizing':
        scale_factor = np.std(y_pred) * 3  # Scale so 3-sigma pred = full position
    
    positions = position_from_pred(y_pred, policy=policy, threshold=threshold, 
                                   scale_factor=scale_factor or 1.0)
    
    # 2. Apply volatility targeting if requested
    vol_scalar = None
    if apply_vol_targeting:
        if returns_for_vol is None:
            # Use actual returns if no separate vol series provided
            returns_for_vol = y_true
        positions, vol_scalar = apply_volatility_targeting(
            positions, returns_for_vol,
            target_vol=target_vol, vol_lookback=vol_lookback,
            max_leverage=max_leverage, min_leverage=min_leverage,
            frequency=frequency
        )
    
    # 3. Compute strategy returns
    strategy_returns = compute_strategy_returns(
        positions, y_true,
        transaction_cost=transaction_cost, apply_tc=apply_tc
    )
    
    # 4. Compute equity curve
    equity_curve = compute_equity_curve(strategy_returns)
    
    # 5. Compute metrics
    n_periods = len(strategy_returns)
    
    # Total return
    total_return = equity_curve[-1] - 1.0 if len(equity_curve) > 0 else 0.0
    
    # Mean and std of returns
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    # Annualized metrics
    ann_return = mean_return * ann_factor
    ann_volatility = std_return * np.sqrt(ann_factor)
    
    # Sharpe ratio
    if std_return == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_return / std_return) * np.sqrt(ann_factor)
    
    # Max drawdown
    max_dd = compute_max_drawdown(equity_curve)
    
    # Hit rate (directional accuracy)
    hit_rate = calculate_hit_rate(y_true, y_pred)
    
    # Information coefficient
    ic = calculate_ic(y_true, y_pred)
    
    # Decile spread
    decile_spread = calculate_decile_spread(y_true, y_pred)
    
    # Trade statistics
    active_mask = positions != 0
    trade_count = np.sum(active_mask)
    holding_frequency = trade_count / n_periods if n_periods > 0 else 0.0
    
    # Average return per trade (when position != 0)
    if trade_count > 0:
        avg_return_per_trade = np.mean(strategy_returns[active_mask])
    else:
        avg_return_per_trade = 0.0
    
    # Win rate (fraction of positive returns when trading)
    if trade_count > 0:
        win_rate = np.mean(strategy_returns[active_mask] > 0)
    else:
        win_rate = 0.0
    
    return {
        'policy': policy,
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'hit_rate': hit_rate,
        'win_rate': win_rate,
        'ic': ic,
        'decile_spread': decile_spread,
        'trade_count': int(trade_count),
        'holding_frequency': holding_frequency,
        'avg_return_per_trade': avg_return_per_trade,
        'annualization_factor': ann_factor,
        'vol_targeting_applied': apply_vol_targeting,
        'n_periods': n_periods
    }


def evaluate_all_policies(y_true, y_pred, threshold=0.03,
                          returns_for_vol=None, apply_vol_targeting=False,
                          target_vol=DEFAULT_TARGET_VOL, vol_lookback=DEFAULT_VOL_LOOKBACK,
                          max_leverage=DEFAULT_MAX_LEVERAGE, min_leverage=DEFAULT_MIN_LEVERAGE,
                          transaction_cost=0.0, apply_tc=False,
                          frequency=None):
    """
    Evaluate all trading policies and return comparison table.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        threshold: Threshold for thresholded policy
        returns_for_vol: Returns for vol calculation
        apply_vol_targeting: Whether to apply vol targeting
        target_vol: Target vol for overlay
        vol_lookback: Vol lookback window
        max_leverage: Max leverage
        min_leverage: Min leverage
        transaction_cost: Transaction cost
        apply_tc: Apply transaction costs
        frequency: Data frequency
        
    Returns:
        Dictionary mapping policy name to metrics dict
    """
    results = {}
    
    for policy in TRADING_POLICIES:
        results[policy] = evaluate_policy(
            y_true, y_pred, policy=policy, threshold=threshold,
            returns_for_vol=returns_for_vol, apply_vol_targeting=apply_vol_targeting,
            target_vol=target_vol, vol_lookback=vol_lookback,
            max_leverage=max_leverage, min_leverage=min_leverage,
            transaction_cost=transaction_cost, apply_tc=apply_tc,
            frequency=frequency
        )
    
    return results


def print_policy_comparison(results, title="Strategy Policy Comparison"):
    """
    Print a formatted comparison of policy evaluation results.
    
    Args:
        results: Dictionary from evaluate_all_policies
        title: Title for the table
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # Header
    header = f"{'Policy':<20} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'HitRate':>8} {'IC':>8}"
    print(header)
    print("-" * 70)
    
    for policy, metrics in results.items():
        vol_tag = " (VT)" if metrics.get('vol_targeting_applied', False) else ""
        policy_name = f"{policy}{vol_tag}"
        print(f"{policy_name:<20} {metrics['sharpe']:>8.2f} "
              f"{metrics['total_return']:>9.1%} {metrics['max_drawdown']:>8.1%} "
              f"{metrics['hit_rate']:>8.1%} {metrics['ic']:>8.3f}")
    
    print(f"{'='*70}\n")


# ============================================================================
# Threshold Tuning (Anti-Policy-Overfit)
# ============================================================================

def generate_threshold_grid(y_pred, n_percentiles=10, min_threshold=0.0, max_threshold=None):
    """
    Generate a grid of threshold values based on prediction percentiles.
    
    This uses percentiles of |pred| to create a data-driven threshold grid,
    ensuring thresholds are meaningful relative to the prediction distribution.
    
    Args:
        y_pred: Array of predictions
        n_percentiles: Number of percentile-based thresholds to generate
        min_threshold: Minimum threshold value
        max_threshold: Maximum threshold (if None, uses 95th percentile of |pred|)
        
    Returns:
        Array of threshold values to try
    """
    y_pred = np.array(y_pred)
    abs_pred = np.abs(y_pred)
    
    # Generate percentile-based thresholds
    percentiles = np.linspace(10, 90, n_percentiles)
    threshold_values = [np.percentile(abs_pred, p) for p in percentiles]
    
    # Add some fixed common thresholds for good measure
    fixed_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    
    # Combine and deduplicate
    all_thresholds = sorted(set(threshold_values + fixed_thresholds))
    
    # Filter by bounds
    if max_threshold is None:
        max_threshold = np.percentile(abs_pred, 95)
    
    grid = [t for t in all_thresholds if min_threshold <= t <= max_threshold]
    
    # Ensure we have at least some thresholds
    if len(grid) < 3:
        grid = [0.01, 0.02, 0.03]
    
    return np.array(grid)


def tune_threshold(y_val_true, y_val_pred, 
                   criterion='sharpe',
                   threshold_grid=None,
                   n_grid_points=10,
                   min_trade_fraction=0.1,
                   frequency=None,
                   returns_for_vol=None,
                   apply_vol_targeting=False):
    """
    Tune the threshold τ for the thresholded policy using validation data.
    
    This function performs grid search over threshold values to find the
    optimal τ that maximizes a given criterion on the validation set.
    
    Args:
        y_val_true: Validation actual returns
        y_val_pred: Validation predictions
        criterion: Metric to optimize - one of:
            - 'sharpe': Maximize Sharpe ratio
            - 'ic_spread': Maximize IC * decile_spread (predictive spread)
            - 'total_return': Maximize total return
            - 'hit_rate': Maximize directional accuracy when trading
        threshold_grid: Specific thresholds to try (if None, auto-generated)
        n_grid_points: Number of grid points if auto-generating
        min_trade_fraction: Minimum fraction of periods that must have trades
        frequency: Data frequency for annualization
        returns_for_vol: Returns for volatility targeting (optional)
        apply_vol_targeting: Whether to apply vol targeting during tuning
        
    Returns:
        Dictionary with:
            - 'best_threshold': Optimal threshold value
            - 'best_score': Score achieved at best threshold
            - 'criterion': Criterion used for optimization
            - 'grid_results': Full results for all thresholds tried
            - 'n_thresholds_tried': Number of thresholds evaluated
    """
    y_val_true = np.array(y_val_true)
    y_val_pred = np.array(y_val_pred)
    n_samples = len(y_val_true)
    
    # Generate threshold grid if not provided
    if threshold_grid is None:
        threshold_grid = generate_threshold_grid(y_val_pred, n_percentiles=n_grid_points)
    
    # Evaluate each threshold
    grid_results = []
    best_threshold = threshold_grid[0]
    best_score = -np.inf
    
    for tau in threshold_grid:
        # Evaluate thresholded policy at this threshold
        result = evaluate_policy(
            y_val_true, y_val_pred,
            policy='thresholded',
            threshold=tau,
            returns_for_vol=returns_for_vol,
            apply_vol_targeting=apply_vol_targeting,
            frequency=frequency
        )
        
        # Skip if too few trades (policy overfit to no-trade)
        if result['holding_frequency'] < min_trade_fraction:
            score = -np.inf
        else:
            # Compute score based on criterion
            if criterion == 'sharpe':
                score = result['sharpe']
            elif criterion == 'ic_spread':
                # IC * decile_spread captures both ranking quality and monetization
                score = result['ic'] * result['decile_spread']
            elif criterion == 'total_return':
                score = result['total_return']
            elif criterion == 'hit_rate':
                score = result['win_rate']  # Use win_rate (when trading) not hit_rate
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
        
        grid_results.append({
            'threshold': tau,
            'score': score,
            'sharpe': result['sharpe'],
            'total_return': result['total_return'],
            'hit_rate': result['hit_rate'],
            'win_rate': result['win_rate'],
            'ic': result['ic'],
            'decile_spread': result['decile_spread'],
            'trade_count': result['trade_count'],
            'holding_frequency': result['holding_frequency']
        })
        
        if score > best_score:
            best_score = score
            best_threshold = tau
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'criterion': criterion,
        'grid_results': grid_results,
        'n_thresholds_tried': len(threshold_grid)
    }


def tune_and_evaluate_fold(y_val_true, y_val_pred, y_test_true, y_test_pred,
                           criterion='sharpe',
                           threshold_grid=None,
                           n_grid_points=10,
                           min_trade_fraction=0.1,
                           frequency=None,
                           returns_for_vol_val=None,
                           returns_for_vol_test=None,
                           apply_vol_targeting=False):
    """
    Tune threshold on validation set and evaluate on test set.
    
    This is the main function for per-fold threshold tuning to avoid
    policy overfit. The threshold is selected purely on validation data,
    then applied to the test set without peeking.
    
    Args:
        y_val_true: Validation actual returns
        y_val_pred: Validation predictions
        y_test_true: Test actual returns  
        y_test_pred: Test predictions
        criterion: Optimization criterion for tuning
        threshold_grid: Specific thresholds to try
        n_grid_points: Grid points for auto-generation
        min_trade_fraction: Minimum trade fraction constraint
        frequency: Data frequency
        returns_for_vol_val: Val period returns for vol targeting
        returns_for_vol_test: Test period returns for vol targeting
        apply_vol_targeting: Whether to apply vol targeting
        
    Returns:
        Dictionary with:
            - 'tuned_threshold': The threshold selected on validation
            - 'val_metrics': Metrics on validation at tuned threshold
            - 'test_metrics': Metrics on test at tuned threshold
            - 'tuning_details': Full tuning grid results
    """
    # Step 1: Tune on validation
    tuning_result = tune_threshold(
        y_val_true, y_val_pred,
        criterion=criterion,
        threshold_grid=threshold_grid,
        n_grid_points=n_grid_points,
        min_trade_fraction=min_trade_fraction,
        frequency=frequency,
        returns_for_vol=returns_for_vol_val,
        apply_vol_targeting=apply_vol_targeting
    )
    
    best_tau = tuning_result['best_threshold']
    
    # Step 2: Evaluate validation at best threshold (for reference)
    val_metrics = evaluate_policy(
        y_val_true, y_val_pred,
        policy='thresholded',
        threshold=best_tau,
        returns_for_vol=returns_for_vol_val,
        apply_vol_targeting=apply_vol_targeting,
        frequency=frequency
    )
    
    # Step 3: Evaluate test at best threshold (the real evaluation)
    test_metrics = evaluate_policy(
        y_test_true, y_test_pred,
        policy='thresholded',
        threshold=best_tau,
        returns_for_vol=returns_for_vol_test,
        apply_vol_targeting=apply_vol_targeting,
        frequency=frequency
    )
    
    return {
        'tuned_threshold': best_tau,
        'tuning_score': tuning_result['best_score'],
        'tuning_criterion': criterion,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'tuning_details': tuning_result
    }


def aggregate_tuned_policy_results(fold_results, include_baseline=True):
    """
    Aggregate results from multiple folds with per-fold threshold tuning.
    
    Args:
        fold_results: List of dictionaries from tune_and_evaluate_fold
        include_baseline: Whether to include baseline (fixed threshold) comparison
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not fold_results:
        return {}
    
    # Extract per-fold values
    thresholds = [f['tuned_threshold'] for f in fold_results]
    val_sharpes = [f['val_metrics']['sharpe'] for f in fold_results]
    test_sharpes = [f['test_metrics']['sharpe'] for f in fold_results]
    test_returns = [f['test_metrics']['total_return'] for f in fold_results]
    test_hit_rates = [f['test_metrics']['hit_rate'] for f in fold_results]
    test_trade_counts = [f['test_metrics']['trade_count'] for f in fold_results]
    test_ics = [f['test_metrics']['ic'] for f in fold_results]
    
    # Compute aggregate metrics
    result = {
        'n_folds': len(fold_results),
        'threshold_mean': np.mean(thresholds),
        'threshold_std': np.std(thresholds),
        'threshold_min': np.min(thresholds),
        'threshold_max': np.max(thresholds),
        'thresholds_per_fold': thresholds,
        
        'val_sharpe_mean': np.mean(val_sharpes),
        'val_sharpe_std': np.std(val_sharpes),
        
        'test_sharpe_mean': np.mean(test_sharpes),
        'test_sharpe_std': np.std(test_sharpes),
        'test_sharpe_median': np.median(test_sharpes),
        
        'test_return_mean': np.mean(test_returns),
        'test_return_sum': np.sum(test_returns),  # Approx total return
        
        'test_hit_rate_mean': np.mean(test_hit_rates),
        'test_trade_count_total': np.sum(test_trade_counts),
        'test_ic_mean': np.mean(test_ics),
        
        # Detailed per-fold results for logging
        'fold_details': fold_results
    }
    
    return result


def print_threshold_tuning_summary(agg_results, title="Threshold-Tuned Policy Results"):
    """
    Print a formatted summary of threshold tuning results.
    
    Args:
        agg_results: Dictionary from aggregate_tuned_policy_results
        title: Title for the summary
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    print(f"\nThreshold Statistics (across {agg_results['n_folds']} folds):")
    print(f"  Mean τ:     {agg_results['threshold_mean']:.4f}")
    print(f"  Std τ:      {agg_results['threshold_std']:.4f}")
    print(f"  Range:      [{agg_results['threshold_min']:.4f}, {agg_results['threshold_max']:.4f}]")
    
    print(f"\nValidation Performance (used for tuning):")
    print(f"  Sharpe:     {agg_results['val_sharpe_mean']:.3f} ± {agg_results['val_sharpe_std']:.3f}")
    
    print(f"\nTest Performance (out-of-sample):")
    print(f"  Sharpe:     {agg_results['test_sharpe_mean']:.3f} ± {agg_results['test_sharpe_std']:.3f}")
    print(f"  Hit Rate:   {agg_results['test_hit_rate_mean']:.1%}")
    print(f"  IC:         {agg_results['test_ic_mean']:.3f}")
    print(f"  Total Trades: {agg_results['test_trade_count_total']}")
    print(f"  Approx Return: {agg_results['test_return_sum']:.2%}")
    
    print(f"\nPer-Fold Thresholds: {[f'{t:.4f}' for t in agg_results['thresholds_per_fold']]}")
    print(f"{'='*70}\n")


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================

def calculate_strategy_metrics(y_true, y_pred, pred_clip=None, frequency=None):
    """
    [LEGACY] Simulates a simple Long/Short strategy based on prediction sign.
    
    This function is kept for backward compatibility.
    For new code, use evaluate_policy(policy='long_short').
    """
    y_pred_strat = np.array(y_pred)
    if pred_clip is not None:
        y_pred_strat = np.clip(y_pred_strat, -pred_clip, pred_clip)
    
    return evaluate_policy(
        y_true, y_pred_strat, policy='long_short',
        frequency=frequency
    )


def calculate_bigmove_strategy_metrics(y_true, y_pred, threshold=0.03, pred_clip=None, frequency=None):
    """
    [LEGACY] Simulates a strategy that only trades when a big move is predicted.
    
    This function is kept for backward compatibility.
    For new code, use evaluate_policy(policy='thresholded').
    """
    y_pred_strat = np.array(y_pred)
    if pred_clip is not None:
        y_pred_strat = np.clip(y_pred_strat, -pred_clip, pred_clip)
    
    return evaluate_policy(
        y_true, y_pred_strat, policy='thresholded', threshold=threshold,
        frequency=frequency
    )


def calculate_tail_metrics(y_true, y_pred, threshold=0.05):
    """
    Calculates Precision and Recall for 'Big Shifts' (> threshold).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Masks for Actual Extremes
    actual_up_mask = y_true > threshold
    actual_down_mask = y_true < -threshold
    
    # Masks for Predicted Extremes
    pred_up_mask = y_pred > threshold
    pred_down_mask = y_pred < -threshold
    
    metrics = {}
    
    # 1. Precision: When model predicts Big Move, is it right?
    if np.sum(pred_up_mask) > 0:
        correct_up = np.sum(actual_up_mask & pred_up_mask)
        metrics['precision_up_strict'] = correct_up / np.sum(pred_up_mask)
    else:
        metrics['precision_up_strict'] = 0.0
        
    if np.sum(pred_down_mask) > 0:
        correct_down = np.sum(actual_down_mask & pred_down_mask)
        metrics['precision_down_strict'] = correct_down / np.sum(pred_down_mask)
    else:
        metrics['precision_down_strict'] = 0.0
        
    # 2. Recall: When Big Move happens, did model predict it?
    if np.sum(actual_up_mask) > 0:
        caught_up = np.sum(actual_up_mask & pred_up_mask)
        metrics['recall_up_strict'] = caught_up / np.sum(actual_up_mask)
    else:
        metrics['recall_up_strict'] = 0.0
        
    if np.sum(actual_down_mask) > 0:
        caught_down = np.sum(actual_down_mask & pred_down_mask)
        metrics['recall_down_strict'] = caught_down / np.sum(actual_down_mask)
    else:
        metrics['recall_down_strict'] = 0.0
        
    metrics['count_actual_up'] = int(np.sum(actual_up_mask))
    metrics['count_actual_down'] = int(np.sum(actual_down_mask))
    metrics['count_pred_up'] = int(np.sum(pred_up_mask))
    metrics['count_pred_down'] = int(np.sum(pred_down_mask))
    
    return metrics


# ============================================================================
# Summary Function for Walk-Forward Results
# ============================================================================

def evaluate_walkforward_results(actuals, predictions, daily_returns=None,
                                 threshold=0.03, frequency=None,
                                 apply_vol_targeting=False):
    """
    Comprehensive evaluation of walk-forward predictions.
    
    Args:
        actuals: Array of actual returns
        predictions: Array of predicted returns
        daily_returns: Daily returns for vol targeting (optional)
        threshold: Threshold for thresholded policy
        frequency: Data frequency
        apply_vol_targeting: Whether to apply vol targeting
        
    Returns:
        Dictionary with all metrics across all policies
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    # Basic metrics
    basic_metrics = {
        'ic': calculate_ic(actuals, predictions),
        'hit_rate': calculate_hit_rate(actuals, predictions),
        'decile_spread': calculate_decile_spread(actuals, predictions),
        'n_samples': len(actuals)
    }
    
    # Tail metrics
    tail_metrics = calculate_tail_metrics(actuals, predictions, threshold=threshold)
    
    # Evaluate all policies
    policy_results = evaluate_all_policies(
        actuals, predictions, threshold=threshold,
        returns_for_vol=daily_returns,
        apply_vol_targeting=apply_vol_targeting,
        frequency=frequency
    )
    
    # Also evaluate with vol targeting if not already applied
    if not apply_vol_targeting and daily_returns is not None:
        policy_results_vt = evaluate_all_policies(
            actuals, predictions, threshold=threshold,
            returns_for_vol=daily_returns,
            apply_vol_targeting=True,
            frequency=frequency
        )
        # Add vol-targeted results with "_vt" suffix
        for policy, metrics in policy_results_vt.items():
            policy_results[f"{policy}_vt"] = metrics
    
    return {
        'basic': basic_metrics,
        'tail': tail_metrics,
        'policies': policy_results
    }
