import sys
import os
import pandas as pd
import numpy as np
import collections

# Add parent directory to path to allow imports from ML
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import config, train_walkforward

def run_benchmark(models_to_run=None):
    """
    Run the benchmark suite for specified models.
    """
    if models_to_run is None:
        models_to_run = ['Ridge', 'XGBoost', 'Ridge_Residual_XGB']
        
    print(f"🚀 Starting Benchmark Suite for: {models_to_run}")
    print("=" * 80)
    
    results = collections.defaultdict(dict)
    
    # Store original model type to restore later
    original_model_type = config.MODEL_TYPE
    
    try:
        for model_name in models_to_run:
            print(f"\n\n👉 Running Model: {model_name}")
            print("-" * 50)
            
            # 1. Configure
            config.MODEL_TYPE = model_name
            
            # 2. Run
            # We must ensure we're using the same split logic. 
            # train_walkforward uses the config globally, so it should be fine.
            # However, we should be careful about internal state.
            # Ideally, we'd reload the module or re-init the main components, 
            # but main() creates fresh objects (Logger, Splitter, etc.) so it's okay.
            
            run_output = train_walkforward.main()
            
            # 3. Capture Metrics
            # We want: 
            # - Test IC (mean/std)
            # - Decile spread
            # - Sharpe, MaxDD
            # - Regime conditional stats
            
            # Extract specific metrics directly from the run output dictionaries for precision
            metrics_test = run_output['metrics_test']
            strat_metrics = metrics_test.get('strat_metrics', {})
            
            res = {}
            
            # Core Metrics
            res['IC Mean'] = metrics_test['ic'] 
            
            # Get fold-level IC stats
            if 'fold_metrics_list' in run_output:
                fold_metrics = pd.DataFrame(run_output['fold_metrics_list'])
                if 'ic' in fold_metrics.columns:
                    res['IC Mean (Fold)'] = fold_metrics['ic'].mean()
                    res['IC Std (Fold)'] = fold_metrics['ic'].std()
            
            # Monthly Rebal Metrics (from strat_metrics which respects config.EXECUTION_FREQUENCY="monthly")
            res['Sharpe'] = strat_metrics.get('sharpe', np.nan)
            res['Max Drawdown'] = strat_metrics.get('max_drawdown', np.nan)
            res['Annual Return'] = strat_metrics.get('ann_return', np.nan)
            
            res['Decile Spread'] = metrics_test.get('decile_spread', np.nan)
            
            # Regime Metrics
            regime = run_output['regime_metrics']
            if regime and 'breakdown' in regime:
                for r_id, r_met in regime['breakdown'].items():
                    res[f'Regime {r_id} IC'] = r_met.get('ic', np.nan)
                    res[f'Regime {r_id} Sharpe'] = r_met.get('sharpe', np.nan)
            
            results[model_name] = res
            
    finally:
        config.MODEL_TYPE = original_model_type
        
    # 4. Generate Report
    print("\n\n📊 Generating Consolidated Report...")
    print("=" * 80)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate Deltas
    # Delta vs Ridge
    if 'Ridge' in df_results.columns:
        for col in df_results.columns:
            if col != 'Ridge':
                df_results[f'{col} Delta vs Ridge'] = df_results[col] - df_results['Ridge']

    # Delta vs XGBoost
    if 'XGBoost' in df_results.columns:
        for col in df_results.columns:
            if col != 'XGBoost' and 'Delta' not in col:
                 df_results[f'{col} Delta vs XGB'] = df_results[col] - df_results['XGBoost']
                 
    # Reorder columns: Ridge, XGB, Residual, Deltas
    cols = models_to_run + [c for c in df_results.columns if c not in models_to_run]
    df_results = df_results[cols]
    
    # Print Markdown (fallback to string if tabulate missing)
    try:
        print(df_results.to_markdown())
        md_table = df_results.to_markdown()
    except ImportError:
        print(df_results.to_string())
        md_table = df_results.to_string()
    
    # Save to file
    report_path = os.path.join(config.REPO_ROOT, "Benchmark_Report.md")
    with open(report_path, "w") as f:
        f.write(f"# Benchmark Suite Report\n\n")
        f.write(f"Models: {', '.join(models_to_run)}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n\n")
        f.write("```\n")
        f.write(md_table)
        f.write("\n```\n")
        
    excel_path = os.path.join(config.REPO_ROOT, "Benchmark_Report.xlsx")
    df_results.to_excel(excel_path)
    
    print(f"\n✅ Reports Saved:\n - {report_path}\n - {excel_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run 1 fold only for testing")
    args = parser.parse_args()
    
    if args.dry_run:
        # Hack to limit folds for dry run
        # We can't easily patch the iterator inside main() without changing code
        # But we can patch config.TUNE_MAX_FOLDS? No, that's for tuning.
        # We might need to monkeypatch the splitter or just let it run.
        # Ideally, main() should accept a debug flag, or config should have a DEBUG_RUN flag.
        print("⚠️ Dry Run not strictly supported by train_walkforward.main() without code change.")
        print("   Running full suite (CTRL+C to stop if needed)")
        
    run_benchmark()
