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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

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
    
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Only count periods where actual != 0
    nonzero_mask = y_true != 0
    if np.sum(nonzero_mask) == 0:
        return 0.0
    
    correct = np.sign(y_pred[nonzero_mask]) == np.sign(y_true[nonzero_mask])
    return np.mean(correct)


def calculate_classification_stats(y_true, y_pred):
    """
    Calculates comprehensive classification metrics for directional prediction.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Dictionary with Accuracy, Precision, Recall, F1, and Confusion Matrix raw counts.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Convert to signs (-1, 1). Ignore 0s for simplicity or treat as neutral.
    # We'll treat > 0 as Positive (1) and <= 0 as Negative (0) for standard binary classification
    # But usually in finance, 0 is rare. Let's use > 0 and < 0.
    
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    
    # We only consider non-zero actuals for evaluation usually, but here we cover all.
    # Positives: > 0
    # Negatives: <= 0 (Treat 0 as negative/neutral)
    
    y_true_bin = (true_sign > 0).astype(int)
    y_pred_bin = (pred_sign > 0).astype(int)
    
    # Calculation
    # TP: Pred=1, True=1
    # FP: Pred=1, True=0
    # TN: Pred=0, True=0
    # FN: Pred=0, True=1
    
    TP = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    FP = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    TN = np.sum((y_pred_bin == 0) & (y_true_bin == 0))
    FN = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    
    total = len(y_true_bin)
    
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    # Precision (Positive): TP / (TP + FP)
    precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall (Positive): TP / (TP + FN)
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # Precision (Negative): TN / (TN + FN)
    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    
    # Recall (Negative): TN / (TN + FP)
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0
    
    return {
        'Accuracy': accuracy,
        'Precision (Up)': precision_pos,
        'Recall (Up)': recall_pos,
        'F1 Score (Up)': f1_pos,
        'Precision (Down)': precision_neg,
        'Recall (Down)': recall_neg,
        'True Positives (TP)': int(TP),
        'False Positives (FP)': int(FP),
        'True Negatives (TN)': int(TN),
        'False Negatives (FN)': int(FN),
        'Total Up Preds': int(TP + FP),
        'Total Down Preds': int(TN + FN),
        'Total Samples': int(total)
    }


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
    # Use the robust analysis function to handle small sample sizes (fallback to quintiles)
    # and ensure consistent logic.
    result = calculate_decile_analysis(y_true, y_pred, n_quantiles=10)
    return result['spread']


def calculate_decile_analysis(y_true, y_pred, n_quantiles=10):
    """
    Comprehensive decile (quantile) analysis of predictions vs actual returns.
    
    This is the key diagnostic for understanding where alpha is concentrated.
    If the model has predictive power, top deciles should have higher returns
    than bottom deciles.
    
    Automatic fallback to Quintiles (5) or Median Split (2) if sample size is too small.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        n_quantiles: Number of quantiles (default 10 for deciles)
        
    Returns:
        Dictionary with:
        - spread: Top decile mean - Bottom decile mean
        - spread_tstat: t-statistic for the spread
        - top_mean: Mean return of top decile
        - bottom_mean: Mean return of bottom decile
        - top_count: Number of samples in top decile
        - bottom_count: Number of samples in bottom decile
        - monotonicity: Spearman correlation of decile ranks vs decile returns
        - quantile_returns: List of mean returns by quantile (lowest to highest pred)
    """
    from scipy.stats import ttest_ind
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    N = len(y_pred)
    
    effective_quantiles = n_quantiles
    
    # --- Robust Usage Check & Fallback ---
    # We ideally want at least 2 samples per bin to calculate a mean, and definitely to do a t-test.
    # If N is small (e.g. < 20 for deciles), we fallback.
    
    if N < effective_quantiles * 2:
        # Not enough samples for requested quantiles. Try fallbacks.
        if N >= 10: # Enough for Quintiles (5 bins * 2 samples)
            print(f"  [Metrics] Info: Sample size {N} too small for {n_quantiles} bins. Falling back to Quintiles (5).")
            effective_quantiles = 5
        elif N >= 4: # Enough for Median Split (2 bins * 2 samples)
            print(f"  [Metrics] Info: Sample size {N} too small for {n_quantiles} bins. Falling back to Median Split (2).")
            effective_quantiles = 2
        else:
            # Too few samples to calculate spread meaningfuly
            return {
                'spread': 0.0, 'spread_tstat': 0.0, 'spread_pvalue': 1.0,
                'top_mean': 0.0, 'bottom_mean': 0.0,
                'top_count': 0, 'bottom_count': 0,
                'monotonicity': 0.0, 'quantile_returns': []
            }
            
    # Assign quantile labels based on predictions
    try:
        # Method='first' ensures we force bins even if values are identical (handled by rank usually, but qcut handles it)
        # duplicates='drop' merges bins if edges are unique. We want to avoid crashing.
        quantile_labels = pd.qcut(y_pred, effective_quantiles, labels=False, duplicates='drop')
    except ValueError:
        # Not enough unique values for quantiles
        return {
            'spread': 0.0, 'spread_tstat': 0.0, 'spread_pvalue': 1.0,
            'top_mean': 0.0, 'bottom_mean': 0.0,
            'top_count': 0, 'bottom_count': 0,
            'monotonicity': 0.0, 'quantile_returns': []
        }
    
    n_actual_quantiles = len(np.unique(quantile_labels))
    
    # Compute mean return per quantile
    quantile_returns = []
    for q in range(n_actual_quantiles):
        mask = quantile_labels == q
        if np.sum(mask) > 0:
            quantile_returns.append(np.mean(y_true[mask]))
        else:
            quantile_returns.append(0.0) # Handle empty bin gracefully
    
    # Top and bottom buckets
    # Note: If duplicates dropped bins, n_actual might be < effective. 
    # The highest label is n_actual_quantiles - 1.
    top_mask = quantile_labels == (n_actual_quantiles - 1)
    bottom_mask = quantile_labels == 0
    
    top_returns = y_true[top_mask]
    bottom_returns = y_true[bottom_mask]
    
    top_mean = np.mean(top_returns) if len(top_returns) > 0 else 0.0
    bottom_mean = np.mean(bottom_returns) if len(bottom_returns) > 0 else 0.0
    spread = top_mean - bottom_mean
    
    # T-test for spread significance
    # We need sufficient samples in top/bottom buckets
    if len(top_returns) > 1 and len(bottom_returns) > 1:
        # For small N with unequal variances, Welch's t-test (equal_var=False) is safer, 
        # but standard t-test is acceptable here.
        with np.errstate(all='ignore'): # Suppress divide by zero warnings in ttest
             tstat, pvalue = ttest_ind(top_returns, bottom_returns)
             if np.isnan(tstat): tstat, pvalue = 0.0, 1.0
    else:
        tstat, pvalue = 0.0, 1.0
    
    # Monotonicity: correlation between quantile rank and mean return
    # High monotonicity = returns increase smoothly with prediction rank
    valid_returns = [r for r in quantile_returns if not np.isnan(r)]
    if len(valid_returns) >= 3:
        mono_corr, _ = spearmanr(range(len(valid_returns)), valid_returns)
        monotonicity = mono_corr if not np.isnan(mono_corr) else 0.0
    elif len(valid_returns) == 2:
        # For 2 bins, monotonicity is simply sign of spread (1.0 or -1.0)
        monotonicity = 1.0 if valid_returns[1] > valid_returns[0] else -1.0
    else:
        monotonicity = 0.0
    
    return {
        'spread': spread,
        'spread_tstat': tstat,
        'spread_pvalue': pvalue,
        'top_mean': top_mean,
        'bottom_mean': bottom_mean,
        'top_count': int(np.sum(top_mask)),
        'bottom_count': int(np.sum(bottom_mask)),
        'monotonicity': monotonicity,
        'quantile_returns': quantile_returns
    }


def calculate_coverage_performance(y_true, y_pred, thresholds=None, frequency=None, 
                                   dates=None, execution_frequency=None):
    """
    Calculate coverage (% periods traded) vs performance (Sharpe) for various thresholds.
    
    This reveals the coverage-performance tradeoff: higher thresholds mean
    fewer trades but potentially higher quality signals.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        thresholds: List of threshold values to evaluate (if None, auto-generated)
        frequency: Data frequency for annualization
        dates: Optional dates for monthly execution mode
        execution_frequency: Execution frequency mode
        
    Returns:
        Dictionary with:
        - threshold_analysis: List of {threshold, coverage, sharpe, return, ic} dicts
        - best_sharpe_threshold: Threshold with highest Sharpe
        - best_sharpe_coverage: Coverage at best Sharpe threshold
        - coverage_performance_corr: Correlation between coverage and Sharpe
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Auto-generate thresholds if not provided
    if thresholds is None:
        abs_pred = np.abs(y_pred)
        # Use percentiles of |predictions| as thresholds
        thresholds = [0.0]  # Always include 0 (full coverage)
        for pct in [25, 50, 60, 70, 75, 80, 85, 90, 95]:
            thresholds.append(np.percentile(abs_pred, pct))
        thresholds = sorted(set(thresholds))
    
    threshold_analysis = []
    
    for thresh in thresholds:
        result = evaluate_policy(
            y_true, y_pred,
            policy='thresholded',
            threshold=thresh,
            frequency=frequency,
            dates=dates,
            execution_frequency=execution_frequency
        )
        
        coverage = result['holding_frequency']
        sharpe = result['sharpe']
        total_return = result['total_return']
        trade_count = result['trade_count']
        
        # IC only for traded periods
        traded_mask = np.abs(y_pred) > thresh
        if np.sum(traded_mask) > 5:
            traded_ic = calculate_ic(y_true[traded_mask], y_pred[traded_mask])
        else:
            traded_ic = 0.0
        
        threshold_analysis.append({
            'threshold': thresh,
            'coverage': coverage,
            'sharpe': sharpe,
            'total_return': total_return,
            'trade_count': trade_count,
            'ic_when_trading': traded_ic
        })
    
    # Find best Sharpe threshold (excluding 0 coverage)
    valid_analyses = [a for a in threshold_analysis if a['coverage'] > 0.05]
    if valid_analyses:
        best = max(valid_analyses, key=lambda x: x['sharpe'])
        best_sharpe_threshold = best['threshold']
        best_sharpe_coverage = best['coverage']
        best_sharpe = best['sharpe']
    else:
        best_sharpe_threshold = 0.0
        best_sharpe_coverage = 1.0
        best_sharpe = 0.0
    
    # Coverage vs Sharpe correlation
    if len(threshold_analysis) >= 3:
        coverages = [a['coverage'] for a in threshold_analysis]
        sharpes = [a['sharpe'] for a in threshold_analysis]
        if np.std(coverages) > 0 and np.std(sharpes) > 0:
            cov_perf_corr, _ = spearmanr(coverages, sharpes)
        else:
            cov_perf_corr = 0.0
    else:
        cov_perf_corr = 0.0
    
    return {
        'threshold_analysis': threshold_analysis,
        'best_sharpe_threshold': best_sharpe_threshold,
        'best_sharpe_coverage': best_sharpe_coverage,
        'best_sharpe': best_sharpe,
        'coverage_performance_corr': cov_perf_corr if not np.isnan(cov_perf_corr) else 0.0
    }


def calculate_signal_concentration(y_true, y_pred, frequency=None, dates=None, execution_frequency=None):
    """
    Comprehensive signal concentration analysis combining decile spread and coverage.
    
    This is the key diagnostic for understanding if alpha is concentrated in
    confident predictions - the hallmark of a real signal vs noise.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        frequency: Data frequency
        dates: Optional dates for monthly execution mode
        execution_frequency: Execution frequency mode
        
    Returns:
        Dictionary with all signal concentration metrics
    """
    # Decile analysis
    decile = calculate_decile_analysis(y_true, y_pred)
    
    # Coverage-performance analysis
    coverage = calculate_coverage_performance(
        y_true, y_pred, frequency=frequency,
        dates=dates, execution_frequency=execution_frequency
    )
    
    # Basic metrics
    ic = calculate_ic(y_true, y_pred)
    hit_rate = calculate_hit_rate(y_true, y_pred)
    
    return {
        'ic': ic,
        'hit_rate': hit_rate,
        'decile_spread': decile['spread'],
        'decile_spread_tstat': decile['spread_tstat'],
        'decile_spread_pvalue': decile['spread_pvalue'],
        'decile_monotonicity': decile['monotonicity'],
        'top_decile_mean': decile['top_mean'],
        'bottom_decile_mean': decile['bottom_mean'],
        'quantile_returns': decile['quantile_returns'],
        'best_threshold': coverage['best_sharpe_threshold'],
        'best_threshold_coverage': coverage['best_sharpe_coverage'],
        'best_threshold_sharpe': coverage['best_sharpe'],
        'coverage_sharpe_corr': coverage['coverage_performance_corr'],
        'coverage_analysis': coverage['threshold_analysis']
    }


def calculate_regime_metrics(y_true, y_pred, regimes, regime_col_name="Regime"):
    """
    Calculate performance metrics conditional on regime.

    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        regimes: Array of regime labels (e.g., 0 and 1)
        regime_col_name: Name of the regime column for display

    Returns:
        Dictionary with metrics for each regime.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    regimes = np.array(regimes)
    
    # Filter NaNs from basic inputs (regimes handled separately)
    # We only care about rows where we have both y_true and y_pred
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    # Also require valid regime if it's not nan (but allow nan to be skipped effectively by unique check)
    # Actually, let's filter everything to keep lengths consistent
    # If regime is NaN, we can't categorize it, so likely skip
    if len(regimes) == len(mask):
        mask = mask & np.isfinite(regimes)
        
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    regimes = regimes[mask]
    
    # Identify unique regimes (ignoring NaNs)
    unique_regimes = np.unique(regimes[~np.isnan(regimes)])
    unique_regimes = sorted(unique_regimes)
    
    results = {
        'regime_col': regime_col_name,
        'breakdown': {}
    }
    
    total_samples = len(y_true)
    
    for r in unique_regimes:
        mask = regimes == r
        n_samples = np.sum(mask)
        
        if n_samples < 5:
            continue
            
        r_str = str(int(r)) if float(r).is_integer() else str(r)
        
        # Slice data
        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]
        
        # Compute metrics
        ic = calculate_ic(y_true_r, y_pred_r)
        decile_spread = calculate_decile_spread(y_true_r, y_pred_r)
        hit_rate = calculate_hit_rate(y_true_r, y_pred_r)
        
        # Coverage/Sharpe analysis (simplified)
        # We can reuse evaluate_policy for a quick Sharpe check at 0 threshold (always in)
        # or just compute basic signal quality
        
        results['breakdown'][r_str] = {
            'count': int(n_samples),
            'frequency': n_samples / total_samples,
            'ic': ic,
            'decile_spread': decile_spread,
            'hit_rate': hit_rate,
            'mean_actual': np.mean(y_true_r),
            'mean_pred': np.mean(y_pred_r),
            'std_actual': np.std(y_true_r)
        }
        
    return results


def print_signal_concentration_report(results, title="Signal Concentration Analysis"):
    """
    Print a formatted signal concentration report.
    
    Args:
        results: Dictionary from calculate_signal_concentration
        title: Report title
    """
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    
    # Basic metrics
    print(f"\n  Prediction Quality:")
    print(f"    IC (Spearman):   {results['ic']:+.4f}")
    print(f"    Hit Rate:        {results['hit_rate']:.1%}")
    
    # Decile spread
    print(f"\n  Decile Spread (Top - Bottom):")
    print(f"    Spread:          {results['decile_spread']:+.4f}")
    print(f"    T-statistic:     {results['decile_spread_tstat']:+.2f}")
    print(f"    P-value:         {results['decile_spread_pvalue']:.4f}")
    print(f"    Monotonicity:    {results['decile_monotonicity']:+.3f}")
    print(f"    Top Decile Mean: {results['top_decile_mean']:+.4f}")
    print(f"    Bottom Decile:   {results['bottom_decile_mean']:+.4f}")
    
    # Quantile returns
    if results['quantile_returns']:
        q_rets = results['quantile_returns']
        print(f"\n  Returns by Prediction Quantile (low → high):")
        for i, r in enumerate(q_rets):
            bar = "█" * int(abs(r) * 100) if not np.isnan(r) else ""
            sign = "+" if r >= 0 else ""
            print(f"    Q{i+1}: {sign}{r:.4f} {bar}")
    
    # Coverage analysis
    print(f"\n  Coverage vs Performance (Threshold Policies):")
    print(f"    Best Sharpe:       {results['best_threshold_sharpe']:.2f}")
    print(f"    @ Threshold:       {results['best_threshold']:.4f}")
    print(f"    @ Coverage:        {results['best_threshold_coverage']:.1%}")
    print(f"    Cov-Sharpe Corr:   {results['coverage_sharpe_corr']:+.3f}")
    
    # Coverage breakdown
    if results.get('coverage_analysis'):
        print(f"\n  Threshold | Coverage | Sharpe | IC(traded)")
        print(f"  {'-'*45}")
        for a in results['coverage_analysis']:
            print(f"    {a['threshold']:6.4f} |  {a['coverage']:5.1%}  | {a['sharpe']:+5.2f} |   {a['ic_when_trading']:+.3f}")
    
    print(f"{'='*70}\n")


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


def aggregate_daily_to_monthly_signals(y_pred, dates, lookback_days=5):
    """
    Aggregate daily predictions into monthly signals using last N trading days average.
    
    Args:
        y_pred: Array of daily predictions
        dates: Array or Index of dates (DatetimeIndex or similar)
        lookback_days: Number of last trading days to average (default 5)
        
    Returns:
        monthly_signals: Array of monthly signals (one per month)
        monthly_dates: Array of month-end dates
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    
    df = pd.DataFrame({'pred': y_pred}, index=dates)
    
    # Group by year-month
    df['year_month'] = df.index.to_period('M')
    
    # For each month, take the average of the last lookback_days trading days
    monthly_signals = []
    monthly_dates = []
    
    for year_month, group in df.groupby('year_month'):
        # Get last lookback_days days of the month
        last_days = group.tail(lookback_days)
        if len(last_days) > 0:
            monthly_signal = last_days['pred'].mean()
            monthly_signals.append(monthly_signal)
            # Use the last trading day of the month as the signal date
            monthly_dates.append(last_days.index[-1])
    
    return np.array(monthly_signals), pd.DatetimeIndex(monthly_dates)


def create_monthly_positions(monthly_signals, monthly_dates, policy='long_short', 
                            threshold=0.03, scale_factor=1.0):
    """
    Create monthly positions from monthly signals. One trade per month, held for one month.
    No overlapping positions.
    
    Args:
        monthly_signals: Array of monthly signals
        monthly_dates: DatetimeIndex of month-end dates
        policy: Trading policy
        threshold: Threshold for thresholded policy
        scale_factor: Scale factor for continuous_sizing
        
    Returns:
        monthly_positions: Array of positions (one per month)
    """
    # Compute positions from signals
    positions = position_from_pred(
        monthly_signals, 
        policy=policy, 
        threshold=threshold, 
        scale_factor=scale_factor
    )
    
    return positions


def compute_monthly_returns_from_forward_returns(forward_returns, dates, monthly_dates):
    """
    Compute monthly returns from forward returns for monthly positions.
    
    The signal is generated at month-end (last 5 trading days average), and the position
    is entered at the start of the next month. Since forward_returns are already 21-day
    forward returns, we use the forward return starting from the entry date.
    
    Args:
        forward_returns: Array of 21-day forward returns (one per day)
        dates: DatetimeIndex of daily dates
        monthly_dates: DatetimeIndex of month-end dates (signal generation dates)
        
    Returns:
        monthly_returns: Array of monthly returns (one per month, for the holding period)
    """
    import pandas as pd
    
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    if not isinstance(monthly_dates, pd.DatetimeIndex):
        monthly_dates = pd.to_datetime(monthly_dates)
    
    # Convert forward returns to DataFrame
    df = pd.DataFrame({'return': forward_returns}, index=dates)
    
    monthly_returns = []
    
    for signal_date in monthly_dates:
        # Find the next trading day after signal_date (entry date)
        next_trading_days = df.index[df.index > signal_date]
        if len(next_trading_days) > 0:
            entry_date = next_trading_days[0]
            # Use the forward return starting from entry date
            if entry_date in df.index:
                monthly_return = df.loc[entry_date, 'return']
                monthly_returns.append(monthly_return)
            else:
                monthly_returns.append(0.0)
        else:
            monthly_returns.append(0.0)
    
    return np.array(monthly_returns)


def optimize_continuous_k(y_true, y_pred_z, k_grid=None, execution_frequency='monthly'):
    """
    Tune the multiplier k for the continuous sizing policy:
        position = clip(k * zscore(pred), -1, 1)

    Args:
        y_true: Actual returns
        y_pred_z: Z-scored predictions (pred - mu) / sigma
        k_grid: List of k values to try (default [0.5, 1.0, 1.5])
        execution_frequency: 'daily' or 'monthly' for Sharpe calculation

    Returns:
        Dictionary with 'best_k', 'best_sharpe', and 'results'
    """
    if k_grid is None:
        k_grid = [0.5, 1.0, 1.5]

    best_k = 1.0
    best_sharpe = -float('inf')
    results = []

    y_true_arr = np.array(y_true)
    y_pred_z_arr = np.array(y_pred_z)

    # Calculate annualization factor
    ann_factor = get_annualization_factor(execution_frequency)

    for k in k_grid:
        # Calculate position
        pos = np.clip(k * y_pred_z_arr, -1.0, 1.0)
        
        # Calculate strategy returns
        # Note: This ignores transaction costs for tuning simplicity, but consistent with other areas
        strat_ret = pos * y_true_arr
        
        # Calculate Sharpe
        mean_ret = np.mean(strat_ret)
        std_ret = np.std(strat_ret)
        
        if std_ret > 1e-8:
            sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
        else:
            sharpe = 0.0

        results.append({'k': k, 'sharpe': sharpe})

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_k = k

    return {
        'best_k': best_k,
        'best_sharpe': best_sharpe,
        'results': results
    }


def optimize_regime_cap(y_true, y_pred, regimes, cap_grid=None, execution_frequency='monthly'):
    """
    Tune the position cap for Regime 0:
        position[regime == 0] *= cap

    Args:
        y_true: Actual returns
        y_pred: Base positions (e.g. from continuous policy or sign)
        regimes: Binary array (0=Low Vol/Target Regime, 1=Other)
        cap_grid: List of multipliers to try (default [0.0, 0.5, 1.0])
        execution_frequency: 'daily' or 'monthly' for Sharpe calculation

    Returns:
        Dictionary with 'best_cap', 'best_sharpe', and 'results'
    """
    if cap_grid is None:
        cap_grid = [0.0, 0.5, 1.0]
        
    best_cap = 1.0
    best_sharpe = -float('inf')
    results = []
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    regimes_arr = np.array(regimes)
    
    ann_factor = get_annualization_factor(execution_frequency)
    
    for cap in cap_grid:
        # Apply cap to Regime 0
        pos_capped = y_pred_arr.copy()
        # Ensure mask aligns
        mask = (regimes_arr == 0)
        pos_capped[mask] *= cap
        
        # Calculate strategy returns
        strat_ret = pos_capped * y_true_arr
        
        # Calculate Sharpe
        mean_ret = np.mean(strat_ret)
        std_ret = np.std(strat_ret)
        
        if std_ret > 1e-8:
            sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
        else:
            sharpe = 0.0
            
        results.append({'cap': cap, 'sharpe': sharpe})
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_cap = cap
            
    return {
        'best_cap': best_cap,
        'best_sharpe': best_sharpe,
        'results': results
    }


def evaluate_policy(y_true, y_pred, policy='long_short', threshold=0.03,
                    returns_for_vol=None, apply_vol_targeting=False,
                    target_vol=DEFAULT_TARGET_VOL, vol_lookback=DEFAULT_VOL_LOOKBACK,
                    max_leverage=DEFAULT_MAX_LEVERAGE, min_leverage=DEFAULT_MIN_LEVERAGE,
                    transaction_cost=0.0, apply_tc=False,
                    frequency=None, scale_factor=None, dates=None, 
                    execution_frequency=None):
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
        dates: Optional DatetimeIndex or array of dates (required for monthly execution)
        execution_frequency: "daily" or "monthly" execution mode
        
    Returns:
        Dictionary with comprehensive strategy metrics
    """
    import pandas as pd
    from . import config
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check execution frequency (default to config or "daily")
    if execution_frequency is None:
        execution_frequency = getattr(config, 'EXECUTION_FREQUENCY', 'daily')
    
    # Handle monthly execution mode
    if execution_frequency == "monthly":
        if dates is None:
            raise ValueError("dates parameter required for monthly execution frequency")
        
        # Aggregate daily predictions to monthly signals (last 5 trading days average)
        monthly_signals, monthly_dates = aggregate_daily_to_monthly_signals(
            y_pred, dates, lookback_days=5
        )
        
        # Create monthly positions (one per month, no overlapping)
        if scale_factor is None and policy == 'continuous_sizing':
            scale_factor = np.std(monthly_signals) * 3
        
        monthly_positions = create_monthly_positions(
            monthly_signals, monthly_dates, policy=policy, 
            threshold=threshold, scale_factor=scale_factor or 1.0
        )
        
        # Compute monthly returns from forward returns
        # y_true contains 21-day forward returns, which match the monthly holding period
        monthly_returns = compute_monthly_returns_from_forward_returns(
            y_true, dates, monthly_dates
        )
        
        # Use monthly data for evaluation
        y_true_eval = monthly_returns
        y_pred_eval = monthly_signals
        positions = monthly_positions
        frequency = "monthly"  # Override frequency for annualization
        
        # Note: Volatility targeting on monthly positions would need monthly vol,
        # which is less common, so we skip it in monthly mode for now
        if apply_vol_targeting:
            print("Warning: Volatility targeting not implemented for monthly execution mode")
            apply_vol_targeting = False
    else:
        # Daily execution mode (original behavior)
        y_true_eval = y_true
        y_pred_eval = y_pred
    
    # Get annualization factor
    ann_factor = get_annualization_factor(frequency)
    
    # 1. Compute raw positions from predictions (if not already computed in monthly mode)
    if execution_frequency != "monthly":
        if scale_factor is None and policy == 'continuous_sizing':
            scale_factor = np.std(y_pred) * 3  # Scale so 3-sigma pred = full position
        
        positions = position_from_pred(y_pred, policy=policy, threshold=threshold, 
                                       scale_factor=scale_factor or 1.0)
    
    # 2. Apply volatility targeting if requested (daily mode only)
    vol_scalar = None
    if apply_vol_targeting and execution_frequency != "monthly":
        if returns_for_vol is None:
            # Use actual returns if no separate vol series provided
            returns_for_vol = y_true_eval
        positions, vol_scalar = apply_volatility_targeting(
            positions, returns_for_vol,
            target_vol=target_vol, vol_lookback=vol_lookback,
            max_leverage=max_leverage, min_leverage=min_leverage,
            frequency=frequency
        )
    
    # 3. Compute strategy returns
    strategy_returns = compute_strategy_returns(
        positions, y_true_eval,
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
    
    # Trade statistics
    active_mask = positions != 0
    trade_count = np.sum(active_mask)
    holding_frequency = trade_count / n_periods if n_periods > 0 else 0.0
    
    # --- AUDIT UPDATE ---
    # Metrics computed on Active Periods Only (where positions != 0)
    # This prevents dilution by cash periods (zeros) and focuses on trade quality.
    
    if trade_count > 0:
        # subset for active periods
        y_true_active = y_true_eval[active_mask]
        y_pred_active = y_pred_eval[active_mask]
        strategy_rets_active = strategy_returns[active_mask]
        
        # Hit rate (active only) - Manual calc to ensure same mask
        # We define Hit (Directional Accuracy) as sign(pred) == sign(true)
        # But for strategy, Win Rate (profit > 0) is often what's meant.
        # Let's keep strict Directional Accuracy but on the Active Set.
        # We do NOT drop zeros here to enforce "Same Trade Mask"
        
        # Determine direction matches
        # For Long (1): Hit if y_true > 0
        # For Short (-1): Hit if y_true < 0
        # For Flat (0): masked out
        
        # Recalculate positions active to be safe (though active_mask does this)
        pos_active = positions[active_mask]
        
        # Hit: Position * Return > 0 (Profit) or == 0? 
        # Usually Hit Rate in ML = Direction Match.
        # If I am Long and Return is 0, Sign(1) != Sign(0). Miss.
        hits = np.sign(pos_active) == np.sign(y_true_active)
        hit_rate = np.mean(hits)
        
        # Overwrite the generic calculate_hit_rate call which might drop zeros
        # hit_rate = calculate_hit_rate(y_true_active, y_pred_active)
        
        # IC (active only)
        ic = calculate_ic(y_true_active, y_pred_active)
        
        # Decile spread (active only - likely noisy if count small, but consistent)
        decile_spread = calculate_decile_spread(y_true_active, y_pred_active)
        
        # Win rate (active only)
        win_rate = np.mean(strategy_rets_active > 0)
        
        # Average return per trade
        avg_return_per_trade = np.mean(strategy_rets_active)
        
        # Active Sharpe (Quality of Trades)
        # Note: We use daily returns for Sharpe, not per-trade, so we use strategy_rets_active
        mean_active = np.mean(strategy_rets_active)
        std_active = np.std(strategy_rets_active)
        
        if std_active == 0:
            sharpe = 0.0
        else:
            sharpe = (mean_active / std_active) * np.sqrt(ann_factor)
            
    else:
        # No trades -> Metrics are Undefined (NaN), not Zero
        hit_rate = np.nan
        ic = np.nan
        decile_spread = np.nan
        win_rate = np.nan
        avg_return_per_trade = np.nan
        sharpe = np.nan
        
    # Total metrics (Portfolio Level)
    # Total return and MaxDD still use the full equity curve (including cash/zeros)
    # as these are portfolio constraints.
    
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
                   apply_vol_targeting=False,
                   dates=None,
                   execution_frequency=None):
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
            frequency=frequency,
            dates=dates,
            execution_frequency=execution_frequency
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
                           apply_vol_targeting=False,
                           val_dates=None,
                           test_dates=None,
                           execution_frequency=None):
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
        apply_vol_targeting=apply_vol_targeting,
        dates=val_dates,
        execution_frequency=execution_frequency
    )
    
    best_tau = tuning_result['best_threshold']
    
    # Step 2: Evaluate validation at best threshold (for reference)
    val_metrics = evaluate_policy(
        y_val_true, y_val_pred,
        policy='thresholded',
        threshold=best_tau,
        returns_for_vol=returns_for_vol_val,
        apply_vol_targeting=apply_vol_targeting,
        frequency=frequency,
        dates=val_dates,
        execution_frequency=execution_frequency
    )
    
    # Step 3: Evaluate test at best threshold (the real evaluation)
    test_metrics = evaluate_policy(
        y_test_true, y_test_pred,
        policy='thresholded',
        threshold=best_tau,
        returns_for_vol=returns_for_vol_test,
        apply_vol_targeting=apply_vol_targeting,
        frequency=frequency,
        dates=test_dates,
        execution_frequency=execution_frequency
    )
    

    
    # Debug Logging for Audit (Enhanced)
    print(f"  [Fold Audit] Tuned Threshold: {best_tau:.4f} (Score: {tuning_result['best_score']:.4f})")
    
    val_trades = val_metrics['trade_count']
    val_cov = val_metrics['holding_frequency']
    val_nans = np.isnan(y_val_true).sum()
    print(f"  [Fold Audit] Val: {len(y_val_true)} rows (NaNs: {val_nans}), {val_trades} trades ({val_cov:.1%}) -> Sharpe: {val_metrics['sharpe']:.2f}")
    
    test_trades = test_metrics['trade_count']
    test_cov = test_metrics['holding_frequency']
    test_nans = np.isnan(y_test_true).sum()
    
    if np.isnan(test_metrics['sharpe']):
        sharpe_str = "NaN (No Trades)"
    else:
        sharpe_str = f"{test_metrics['sharpe']:.2f}"
        
    print(f"  [Fold Audit] Test: {len(y_test_true)} rows (NaNs: {test_nans}), {test_trades} trades ({test_cov:.1%}) -> Sharpe: {sharpe_str}")
    
    if test_metrics['trade_count'] == 0:
        print(f"  [Fold Audit] NOTE: No trades in test set at threshold {best_tau:.4f}")
    elif test_metrics['trade_count'] < 5:
        print(f"  [Fold Audit] WARNING: Very low trade count in test ({test_trades}). Metrics might be noisy.")

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
        
        'test_sharpe_mean': np.nanmean(test_sharpes),
        'test_sharpe_std': np.nanstd(test_sharpes),
        'test_sharpe_median': np.nanmedian(test_sharpes),
        
        'test_return_mean': np.nanmean(test_returns),
        'test_return_sum': np.nansum(test_returns),  # Approx total return
        
        'test_hit_rate_mean': np.nanmean(test_hit_rates),
        'test_trade_count_total': np.sum(test_trade_counts),
        'test_ic_mean': np.nanmean(test_ics),
        
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

def calculate_strategy_metrics(y_true, y_pred, pred_clip=None, frequency=None, 
                               dates=None, execution_frequency=None):
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
        frequency=frequency, dates=dates, execution_frequency=execution_frequency
    )


def calculate_bigmove_strategy_metrics(y_true, y_pred, threshold=0.03, pred_clip=None, 
                                      frequency=None, dates=None, execution_frequency=None):
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
        frequency=frequency, dates=dates, execution_frequency=execution_frequency
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
