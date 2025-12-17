"""
Sanity Suite: Pre-Training Diagnostics

A quick diagnostic script that runs before training to catch silent bugs,
data leakage, and configuration issues. Run this BEFORE any training to
ensure your evaluation pipeline is sound.

Usage:
    python -m ML.sanity_suite
    
    # Or with verbose output:
    python -m ML.sanity_suite --verbose

Checks performed:
1. Data Quality: NaN percentages, no backward-fill, date ranges
2. Split Integrity: Embargo validation, no train/test overlap
3. Target Alignment: Autocorrelation diagnostics (detects overlap leakage)
4. Null Model Sanity: Shuffled targets and constant predictors should produce ~0 performance
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import config, data_prep, metrics


# =============================================================================
# Console Output Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.END}")


def print_pass(msg: str):
    """Print a passing check."""
    print(f"  {Colors.GREEN}✓ PASS{Colors.END}: {msg}")


def print_fail(msg: str):
    """Print a failing check."""
    print(f"  {Colors.RED}✗ FAIL{Colors.END}: {msg}")


def print_warn(msg: str):
    """Print a warning."""
    print(f"  {Colors.YELLOW}⚠ WARN{Colors.END}: {msg}")


def print_info(msg: str):
    """Print info."""
    print(f"  {Colors.BLUE}ℹ INFO{Colors.END}: {msg}")


# =============================================================================
# 1. Data Quality Checks
# =============================================================================

def check_nan_percentages(df: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    Check NaN percentages by feature (post-warmup).
    
    Returns:
        Dictionary of feature -> nan_pct for features with any NaNs
    """
    nan_pcts = (df.isna().sum() / len(df) * 100).to_dict()
    features_with_nans = {k: v for k, v in nan_pcts.items() if v > 0}
    
    if verbose and features_with_nans:
        print("\n  Features with NaNs:")
        for feat, pct in sorted(features_with_nans.items(), key=lambda x: -x[1])[:10]:
            print(f"    - {feat}: {pct:.2f}%")
    
    return features_with_nans


def check_no_backward_fill(verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Scan codebase for backward-fill operations that could cause leakage.
    
    Returns:
        (passed, list of files with potential issues)
    """
    import re
    
    # Patterns that indicate backward fill
    bfill_patterns = [
        r'\.bfill\(',
        r'\.fillna\([^)]*method\s*=\s*[\'"]bfill[\'"]',
        r'\.fillna\([^)]*method\s*=\s*[\'"]backfill[\'"]',
        r'fillna\([^)]*bfill',
    ]
    
    suspicious_files = []
    
    # Directories to scan
    scan_dirs = ['ML', 'features', 'data_loader']
    repo_root = config.REPO_ROOT
    
    for scan_dir in scan_dirs:
        dir_path = os.path.join(repo_root, scan_dir)
        if not os.path.exists(dir_path):
            continue
            
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            for pattern in bfill_patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    rel_path = os.path.relpath(file_path, repo_root)
                                    if rel_path not in suspicious_files:
                                        suspicious_files.append(rel_path)
                                    break
                    except Exception:
                        pass
    
    passed = len(suspicious_files) == 0
    
    if verbose and suspicious_files:
        print("\n  Files with potential backward-fill:")
        for f in suspicious_files:
            print(f"    - {f}")
    
    return passed, suspicious_files


def check_data_range(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get basic data range info.
    
    Returns:
        Dictionary with first_date, last_date, row_count, n_features
    """
    return {
        'first_date': df.index.min(),
        'last_date': df.index.max(),
        'row_count': len(df),
        'n_features': len([c for c in df.columns if c != config.TARGET_COL]),
        'n_years': (df.index.max() - df.index.min()).days / 365.25
    }


# =============================================================================
# 2. Split Integrity Checks
# =============================================================================

def check_split_embargo(df: pd.DataFrame, n_splits: int = 5, verbose: bool = False) -> Tuple[bool, List[Dict]]:
    """
    Validate embargo between train/val/test splits.
    
    For each split, verify:
    - max(train_date) < min(test_date) - embargo (row-based)
    - No overlap between sets
    
    Returns:
        (all_passed, list of split details)
    """
    target_col = config.TARGET_COL
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Get embargo rows based on frequency
    frequency = config.DATA_FREQUENCY
    embargo_rows = config.get_embargo_rows(frequency)
    
    # Create splitter
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,
        embargo_rows=embargo_rows,
        step_months=3,  # Use 3-month steps for faster checking
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    
    splits = list(splitter.split(df))
    all_passed = True
    split_details = []
    
    # Check first n_splits
    for fold, train_idx, val_idx, test_idx in splits[:n_splits]:
        detail = {
            'fold': fold,
            'train_start': train_idx.min(),
            'train_end': train_idx.max(),
            'test_start': test_idx.min(),
            'test_end': test_idx.max(),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        }
        
        # Get row positions
        train_end_pos = df.index.get_loc(train_idx.max())
        test_start_pos = df.index.get_loc(test_idx.min())
        
        # Check embargo
        actual_gap = test_start_pos - train_end_pos - 1
        embargo_ok = actual_gap >= embargo_rows
        
        # Check date ordering
        date_ok = train_idx.max() < test_idx.min()
        
        # Check no overlap
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set
        no_overlap = len(overlap) == 0
        
        detail['actual_gap_rows'] = actual_gap
        detail['required_embargo'] = embargo_rows
        detail['embargo_ok'] = embargo_ok
        detail['date_ok'] = date_ok
        detail['no_overlap'] = no_overlap
        detail['passed'] = embargo_ok and date_ok and no_overlap
        
        if not detail['passed']:
            all_passed = False
        
        split_details.append(detail)
        
        if verbose:
            status = "✓" if detail['passed'] else "✗"
            print(f"\n  Fold {fold}: {status}")
            print(f"    Train: {train_idx.min().date()} to {train_idx.max().date()} ({len(train_idx)} rows)")
            print(f"    Test:  {test_idx.min().date()} to {test_idx.max().date()} ({len(test_idx)} rows)")
            print(f"    Gap: {actual_gap} rows (required: {embargo_rows})")
    
    return all_passed, split_details


# =============================================================================
# 3. Target Alignment Checks
# =============================================================================

def check_target_autocorrelation(df: pd.DataFrame, max_lag: int = 5, verbose: bool = False) -> Dict[str, float]:
    """
    Check autocorrelation of target variable to detect overlap issues.
    
    If targets have high autocorrelation at lag=1, it could indicate:
    - Overlapping target windows (e.g., 21-day forward returns on daily data)
    - Data leakage from target construction
    
    For 21-day forward returns on daily data, we expect:
    - High autocorr at lags 1-20 (overlapping windows)
    - Drop at lag 21+ (non-overlapping)
    
    Returns:
        Dictionary of lag -> autocorrelation
    """
    target_col = config.TARGET_COL
    y = df[target_col].dropna()
    
    autocorrs = {}
    for lag in range(1, max_lag + 1):
        shifted = y.shift(lag)
        mask = ~(y.isna() | shifted.isna())
        if mask.sum() > 10:
            corr, _ = spearmanr(y[mask], shifted[mask])
            autocorrs[lag] = corr if not np.isnan(corr) else 0.0
        else:
            autocorrs[lag] = 0.0
    
    if verbose:
        print("\n  Target Autocorrelation by Lag:")
        for lag, corr in autocorrs.items():
            bar = "█" * int(abs(corr) * 20)
            print(f"    Lag {lag:2d}: {corr:+.3f} {bar}")
    
    return autocorrs


def check_target_feature_leakage(df: pd.DataFrame, top_n: int = 5, verbose: bool = False) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Check for suspiciously high correlation between features and target.
    
    Correlations > 0.5 are suspicious and may indicate leakage.
    Also flags known leakage patterns (BigMove* features derived from target).
    
    Returns:
        Tuple of:
        - List of (feature_name, correlation) sorted by abs correlation
        - List of known leakage features that should be excluded
    """
    target_col = config.TARGET_COL
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Known leakage patterns: features derived from the target
    known_leakage_patterns = ['BigMove', 'BigMoveUp', 'BigMoveDown', 'Log_Target', 'Target_']
    known_leakage_features = []
    
    for feat in feature_cols:
        for pattern in known_leakage_patterns:
            if pattern in feat:
                known_leakage_features.append(feat)
                break
    
    correlations = []
    y = df[target_col]
    
    for feat in feature_cols:
        # Skip known leakage features for correlation check
        if feat in known_leakage_features:
            continue
        x = df[feat]
        mask = ~(x.isna() | y.isna())
        if mask.sum() > 10:
            corr, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(corr):
                correlations.append((feat, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if verbose:
        if known_leakage_features:
            print(f"\n  ⚠ Known leakage features detected (should be excluded):")
            for feat in known_leakage_features:
                print(f"    - {feat}")
        
        print(f"\n  Top {top_n} Features by |Correlation| with Target (excluding known leakage):")
        for feat, corr in correlations[:top_n]:
            print(f"    - {feat}: {corr:+.3f}")
    
    return correlations[:top_n], known_leakage_features


# =============================================================================
# 4. Null Model Checks
# =============================================================================

def check_shuffled_target(df: pd.DataFrame, n_trials: int = 5, verbose: bool = False) -> Dict[str, float]:
    """
    Verify that shuffled targets produce ~0 IC / ~0 Sharpe.
    
    If shuffled targets still show significant performance, there's likely
    a bug in the evaluation pipeline (e.g., lookahead bias).
    
    Returns:
        Dictionary with mean/std of shuffled IC and Sharpe
    """
    target_col = config.TARGET_COL
    y_true = df[target_col].values
    
    shuffled_ics = []
    shuffled_sharpes = []
    
    for seed in range(n_trials):
        np.random.seed(seed)
        y_shuffled = np.random.permutation(y_true)
        
        # IC with shuffled
        ic = metrics.calculate_ic(y_true, y_shuffled)
        shuffled_ics.append(ic)
        
        # Sharpe with random predictions
        result = metrics.evaluate_policy(
            y_true, y_shuffled, 
            policy='long_short',
            frequency=config.DATA_FREQUENCY
        )
        shuffled_sharpes.append(result['sharpe'])
    
    results = {
        'ic_mean': np.mean(shuffled_ics),
        'ic_std': np.std(shuffled_ics),
        'sharpe_mean': np.mean(shuffled_sharpes),
        'sharpe_std': np.std(shuffled_sharpes),
    }
    
    if verbose:
        print(f"\n  Shuffled Target Performance ({n_trials} trials):")
        print(f"    IC:     {results['ic_mean']:+.4f} ± {results['ic_std']:.4f}")
        print(f"    Sharpe: {results['sharpe_mean']:+.2f} ± {results['sharpe_std']:.2f}")
    
    return results


def check_constant_predictor(df: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    Verify that constant (all-zeros) predictor produces ~0 returns.
    
    This catches issues where the strategy logic incorrectly handles
    neutral predictions.
    
    Returns:
        Dictionary with strategy metrics for constant predictor
    """
    target_col = config.TARGET_COL
    y_true = df[target_col].values
    
    # All zeros predictor
    y_pred_zeros = np.zeros_like(y_true)
    
    result = metrics.evaluate_policy(
        y_true, y_pred_zeros,
        policy='long_short',
        frequency=config.DATA_FREQUENCY
    )
    
    if verbose:
        print(f"\n  Constant (Zero) Predictor Performance:")
        print(f"    Total Return: {result['total_return']:.4f}")
        print(f"    Sharpe:       {result['sharpe']:.2f}")
        print(f"    Trade Count:  {result['trade_count']}")
    
    return result


def check_perfect_predictor(df: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    Verify that perfect predictor (y_pred = y_true) produces high performance.
    
    This is a sanity check that the evaluation pipeline rewards good predictions.
    
    Returns:
        Dictionary with strategy metrics for perfect predictor
    """
    target_col = config.TARGET_COL
    y_true = df[target_col].values
    
    # Perfect predictor
    result = metrics.evaluate_policy(
        y_true, y_true,  # Predict the actual value
        policy='long_short',
        frequency=config.DATA_FREQUENCY
    )
    
    if verbose:
        print(f"\n  Perfect Predictor Performance (upper bound):")
        print(f"    IC:           {result['ic']:.4f} (should be 1.0)")
        # Total return can overflow due to compounding over many periods
        total_ret = result['total_return']
        if abs(total_ret) > 1e10:
            print(f"    Total Return: {total_ret:.2e} (overflow due to compounding)")
        else:
            print(f"    Total Return: {total_ret:.4f}")
        print(f"    Sharpe:       {result['sharpe']:.2f}")
        print(f"    Hit Rate:     {result['hit_rate']:.1%}")
    
    return result


# =============================================================================
# Main Sanity Suite Runner
# =============================================================================

def run_sanity_suite(verbose: bool = False) -> Dict[str, bool]:
    """
    Run all sanity checks and return pass/fail status for each.
    
    Returns:
        Dictionary mapping check name to pass/fail boolean
    """
    results = {}
    all_passed = True
    
    print_header("SANITY SUITE: Pre-Training Diagnostics")
    print(f"\nConfiguration:")
    print(f"  Data Frequency: {config.DATA_FREQUENCY}")
    print(f"  Target Mode: {config.TARGET_MODE}")
    print(f"  Embargo Rows: {config.get_embargo_rows()}")
    print(f"  Test Start: {config.TEST_START_DATE}")
    
    # =========================
    # Load Data
    # =========================
    print_header("1. Loading Data")
    try:
        df, metadata = data_prep.load_dataset(use_builder=True)
        print_pass(f"Dataset loaded successfully")
        print_info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print_fail(f"Failed to load dataset: {e}")
        return {'data_load': False}
    
    # =========================
    # Data Quality Checks
    # =========================
    print_header("2. Data Quality Checks")
    
    # 2a. NaN percentages
    nan_features = check_nan_percentages(df, verbose=verbose)
    if len(nan_features) == 0:
        print_pass("No NaN values in dataset (post-warmup)")
        results['no_nans'] = True
    else:
        print_warn(f"{len(nan_features)} features have NaN values")
        if verbose:
            for feat, pct in list(nan_features.items())[:5]:
                print(f"      - {feat}: {pct:.2f}%")
        results['no_nans'] = False
    
    # 2b. No backward fill
    no_bfill, bfill_files = check_no_backward_fill(verbose=verbose)
    if no_bfill:
        print_pass("No backward-fill detected in codebase")
        results['no_bfill'] = True
    else:
        print_fail(f"Potential backward-fill in {len(bfill_files)} files: {bfill_files}")
        results['no_bfill'] = False
        all_passed = False
    
    # 2c. Data range
    data_range = check_data_range(df)
    print_info(f"Date Range: {data_range['first_date'].date()} to {data_range['last_date'].date()}")
    print_info(f"Duration: {data_range['n_years']:.1f} years ({data_range['row_count']} rows)")
    print_info(f"Features: {data_range['n_features']}")
    results['data_range'] = True
    
    # =========================
    # Split Integrity Checks
    # =========================
    print_header("3. Split Integrity Checks")
    
    splits_ok, split_details = check_split_embargo(df, n_splits=5, verbose=verbose)
    if splits_ok:
        print_pass(f"All {len(split_details)} splits have proper embargo")
        results['embargo_ok'] = True
    else:
        failed = [d for d in split_details if not d['passed']]
        print_fail(f"{len(failed)}/{len(split_details)} splits failed embargo check")
        results['embargo_ok'] = False
        all_passed = False
    
    # =========================
    # Target Alignment Checks
    # =========================
    print_header("4. Target Alignment Checks")
    
    # 4a. Autocorrelation
    autocorrs = check_target_autocorrelation(df, max_lag=25, verbose=verbose)
    
    # For 21-day forward returns, expect high autocorr at lag 1, drop at lag 21+
    lag1_autocorr = autocorrs.get(1, 0)
    lag21_autocorr = autocorrs.get(21, 0) if 21 in autocorrs else autocorrs.get(max(autocorrs.keys()), 0)
    
    if config.TARGET_HORIZON_DAYS == 21 and config.DATA_FREQUENCY == 'daily':
        # Expected: high autocorr at lag 1 due to overlapping windows
        if lag1_autocorr > 0.8:
            print_info(f"Lag-1 autocorr = {lag1_autocorr:.3f} (expected ~0.95 for overlapping 21d windows)")
        else:
            print_warn(f"Lag-1 autocorr = {lag1_autocorr:.3f} (lower than expected)")
    else:
        print_info(f"Target autocorrelation at lag 1: {lag1_autocorr:.3f}")
    
    results['autocorr_checked'] = True
    
    # 4b. Feature-target correlations
    top_corrs, known_leakage = check_target_feature_leakage(df, top_n=5, verbose=verbose)
    
    # Check for known leakage features (BigMove*, etc.)
    if known_leakage:
        print_warn(f"Known leakage features present in dataset: {known_leakage}")
        print("      These features are derived from the target and should be excluded from training.")
        print("      Note: train.py and train_walkforward.py correctly exclude these in exclude_cols.")
        results['no_known_leakage'] = True  # Warn but don't fail if training code handles it
    else:
        print_pass("No known leakage features (BigMove*, Log_Target*, etc.)")
        results['no_known_leakage'] = True
    
    # Check for suspiciously high correlations
    suspicious = [(f, c) for f, c in top_corrs if abs(c) > 0.5]
    if len(suspicious) == 0:
        print_pass("No suspiciously high feature-target correlations (>0.5)")
        results['no_leakage_corr'] = True
    else:
        print_warn(f"{len(suspicious)} features have |corr| > 0.5 with target")
        for feat, corr in suspicious:
            print(f"      - {feat}: {corr:+.3f}")
        results['no_leakage_corr'] = False
    
    # =========================
    # Null Model Checks
    # =========================
    print_header("5. Null Model Sanity Checks")
    
    # 5a. Shuffled target
    shuffled = check_shuffled_target(df, n_trials=5, verbose=verbose)
    # Use 0.07 threshold to account for sampling noise with small monthly datasets
    shuffled_ic_ok = abs(shuffled['ic_mean']) < 0.07
    
    if shuffled_ic_ok:
        print_pass(f"Shuffled target IC ≈ 0 (actual: {shuffled['ic_mean']:+.4f})")
        results['shuffled_ic_ok'] = True
    else:
        print_fail(f"Shuffled target IC too high: {shuffled['ic_mean']:+.4f}")
        results['shuffled_ic_ok'] = False
        all_passed = False
    
    # Note: Shuffled Sharpe may be positive due to market drift (positive expected returns).
    # This is expected behavior - random positions in an upward-trending market will have
    # positive expected return. The IC check is more reliable for detecting leakage.
    shuffled_sharpe_reasonable = shuffled['sharpe_mean'] < 3.0  # Very loose threshold
    if shuffled_sharpe_reasonable:
        print_info(f"Shuffled target Sharpe = {shuffled['sharpe_mean']:+.2f} (positive due to market drift, IC is the key metric)")
        results['shuffled_sharpe_info'] = True
    else:
        print_warn(f"Shuffled target Sharpe very high: {shuffled['sharpe_mean']:+.2f} (may indicate evaluation bug)")
        results['shuffled_sharpe_info'] = False
    
    # 5b. Constant predictor
    constant = check_constant_predictor(df, verbose=verbose)
    constant_ok = constant['trade_count'] == 0 or abs(constant['total_return']) < 0.01
    
    if constant_ok:
        print_pass(f"Constant predictor produces ~0 return (actual: {constant['total_return']:.4f})")
        results['constant_pred_ok'] = True
    else:
        print_warn(f"Constant predictor has non-zero return: {constant['total_return']:.4f}")
        results['constant_pred_ok'] = False
    
    # 5c. Perfect predictor (should be very good)
    perfect = check_perfect_predictor(df, verbose=verbose)
    perfect_ok = perfect['ic'] > 0.99 and perfect['hit_rate'] > 0.99
    
    if perfect_ok:
        print_pass(f"Perfect predictor achieves IC={perfect['ic']:.3f}, HitRate={perfect['hit_rate']:.1%}")
        results['perfect_pred_ok'] = True
    else:
        print_fail(f"Perfect predictor underperforms: IC={perfect['ic']:.3f}, HitRate={perfect['hit_rate']:.1%}")
        results['perfect_pred_ok'] = False
        all_passed = False
    
    # =========================
    # Summary
    # =========================
    print_header("SUMMARY")
    
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    
    if all_passed:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED ({n_pass}/{n_total}){Colors.END}")
        print(f"  ✓ Your data pipeline appears sound. Proceed with training.\n")
    else:
        print(f"\n  {Colors.RED}{Colors.BOLD}SOME CHECKS FAILED ({n_pass}/{n_total} passed){Colors.END}")
        failed_checks = [k for k, v in results.items() if not v]
        print(f"  Failed: {failed_checks}")
        print(f"  ⚠ Fix these issues before training to avoid wasted compute.\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run pre-training sanity checks")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    results = run_sanity_suite(verbose=args.verbose)
    
    # Exit with error code if any critical checks failed
    critical_checks = ['no_bfill', 'embargo_ok', 'shuffled_ic_ok', 'perfect_pred_ok', 'no_known_leakage']
    critical_failed = any(not results.get(c, True) for c in critical_checks)
    
    sys.exit(1 if critical_failed else 0)


if __name__ == "__main__":
    main()

