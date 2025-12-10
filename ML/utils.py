import os
import datetime
import shutil
from . import config

class ExperimentLogger:
    def __init__(self, model_name="Model", process_tag="Static"):
        # Create base ML_Output directory
        # Use config.REPO_ROOT if available, or calculate it locally to be safe
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = os.path.join(repo_root, 'ML_Output')
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        self.process_tag = process_tag
            
        # Determine Run Number (Ordinal)
        # Scan existing folders to count how many times this model has been run
        existing_runs = [d for d in os.listdir(self.base_dir) if d.startswith(f"{model_name}_")]
        run_count = len(existing_runs) + 1
        
        # Ordinal Suffix (1st, 2nd, 3rd, 4th...)
        if 10 <= run_count % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(run_count % 10, 'th')
            
        run_ordinal = f"{run_count}{suffix}"
        
        # Date Format: Month-Day-Year (e.g., 12-04-2025)
        date_str = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M")
        
        # Construct Folder Name: e.g., MLP_1st_12-04-2025_12-46
        # Adding process_tag (Static/WalkForward) to clarify what kind of run it was
        folder_name = f"{model_name}_{run_ordinal}_{process_tag}_{date_str}"
        
        self.run_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(self.run_dir)
        
        print(f"Experiment logging to: {self.run_dir}")
        
    def log_summary(self, metrics_train, metrics_val, metrics_test, model_type, feature_cols):
        """
        Writes a markdown summary of the run with detailed explanations.
        """
        summary_path = os.path.join(self.run_dir, "summary.md")
        
        # Determine Process Type for Header
        process_tag = self.process_tag
        
        with open(summary_path, 'w') as f:
            f.write(f"# ML Run Summary\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 1. Process Explanation
            f.write(f"## Validation Strategy: {process_tag}\n")
            if "Static" in self.run_dir:
                f.write("> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.\n\n")
            elif "WalkForward" in self.run_dir:
                f.write("> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- **Model**: `{model_type}`\n")
            f.write(f"- **Target Horizon**: {config.TARGET_HORIZON} days (Predicting return 1 month ahead)\n")
            f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
            f.write(f"- **Val Window**: {config.VAL_WINDOW_MONTHS} months\n")
            f.write(f"- **Buffer**: {config.BUFFER_DAYS} days (Embargo to prevent leakage)\n")
            f.write(f"- **Test Start**: {config.TEST_START_DATE}\n\n")
            
            f.write("## Model Description\n")
            if model_type == 'XGBoost':
                f.write("> **XGBoost (Extreme Gradient Boosting)**: A powerful ensemble method that builds a sequence of decision trees. Each new tree corrects the errors of the previous ones. It is known for high performance and speed.\n")
            elif model_type == 'LSTM':
                f.write("> **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) designed for time-series data. Unlike static models, it processes a sequence of past data (e.g., last 10 days) to capture temporal dependencies and trends.\n")
            elif model_type == 'RandomForest':
                f.write("> **Random Forest**: An ensemble of many decision trees trained independently. It reduces overfitting by averaging the predictions of diverse trees.\n")
            elif model_type == 'LinearRegression':
                f.write("> **Linear Regression**: A simple model that assumes a linear relationship between features and the target. Good for establishing a baseline.\n")
            elif model_type == 'MLP':
                f.write("> **MLP (Multi-Layer Perceptron)**: A classic feedforward neural network. It learns non-linear relationships through layers of neurons.\n")
            f.write("\n")
            
            f.write("## Model Parameters\n")
            if model_type == 'RandomForest':
                f.write(f"- N Estimators: {config.RF_N_ESTIMATORS}\n")
                f.write(f"- Max Depth: {config.RF_MAX_DEPTH}\n")
            elif model_type == 'XGBoost':
                f.write(f"- N Estimators: {config.XGB_N_ESTIMATORS}\n")
                f.write(f"- Learning Rate: {config.XGB_LEARNING_RATE}\n")
            elif model_type == 'MLP':
                f.write(f"- Hidden Layers: {config.MLP_HIDDEN_LAYERS}\n")
            elif model_type == 'LinearRegression':
                f.write("- Default sklearn parameters\n")
            f.write("\n")
            
            f.write("## Metrics Explanation\n")
            f.write("### Standard Metrics\n")
            f.write("- **RMSE (Root Mean Squared Error)**: The average magnitude of the prediction error. Lower is better.\n")
            f.write("- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual returns. Lower is better.\n")
            f.write("- **Directional Accuracy**: The percentage of time the model correctly predicted the *sign* (Up/Down) of the return. >50% is the goal.\n")
            
            f.write("### Advanced Metrics\n")
            f.write("- **IC (Information Coefficient)**: The Spearman correlation between predictions and actuals. Measures how well the model ranks returns. >0.05 is good, >0.10 is excellent.\n")
            f.write("- **Strategy Return**: The cumulative return of a simple strategy: Long if Pred > 0, Short if Pred < 0.\n")
            f.write("- **Sharpe Ratio**: Annualized risk-adjusted return of the strategy. >1.0 is good.\n")
            f.write("- **Max Drawdown**: The largest percentage drop from a peak in the strategy's equity curve. Smaller magnitude (closer to 0) is better.\n")
            
            f.write("### Big Shift Analysis (>5%)\n")
            f.write("Focuses on extreme moves (market crashes or rallies) greater than 5% in a month.\n")
            f.write("- **Precision**: When the model predicts a Big Move, how often is it right? (High Precision = Few False Alarms).\n")
            f.write("- **Recall**: When a Big Move actually happens, how often did the model predict it? (High Recall = Few Missed Opportunities).\n\n")
            
            f.write("## Results\n")
            f.write("### Validation (In-Sample / Tuning)\n")
            f.write(f"- RMSE: {metrics_val['rmse']:.6f}\n")
            f.write(f"- MAE: {metrics_val['mae']:.6f}\n")
            f.write(f"- Directional Accuracy: {metrics_val['dir_acc']:.2f}%\n\n")
            
            f.write("### Test (Out of Sample)\n")
            f.write(f"- RMSE: {metrics_test['rmse']:.6f}\n")
            f.write(f"- MAE: {metrics_test['mae']:.6f}\n")
            f.write(f"- Directional Accuracy: {metrics_test['dir_acc']:.2f}%\n")
            
            if 'ic' in metrics_test:
                f.write(f"- IC: {metrics_test['ic']:.4f}\n")
                
            if 'strat_metrics' in metrics_test:
                sm = metrics_test['strat_metrics']
                f.write(f"\n#### Always-In Strategy (Sign-Based)\n")
                f.write(f"- Total Return: {sm['total_return']:.4f}\n")
                f.write(f"- Sharpe Ratio: {sm['sharpe']:.2f}\n")
                f.write(f"- Max Drawdown: {sm['max_drawdown']:.4f}\n")
            
            if 'bigmove_strat' in metrics_test:
                bm = metrics_test['bigmove_strat']
                f.write(f"\n#### Big-Move-Only Strategy\n")
                f.write(f"> Only enters positions when predicted return exceeds threshold.\n\n")
                f.write(f"- Total Return: {bm['total_return']:.4f}\n")
                f.write(f"- Annualized Return: {bm['ann_return']:.4f}\n")
                f.write(f"- Annualized Volatility: {bm['ann_volatility']:.4f}\n")
                f.write(f"- Sharpe Ratio: {bm['sharpe']:.2f}\n")
                f.write(f"- Max Drawdown: {bm['max_drawdown']:.4f}\n")
                f.write(f"- Trade Count: {bm['trade_count']}\n")
                f.write(f"- Holding Frequency: {bm['holding_frequency']:.1%}\n")
                f.write(f"- Avg Return per Trade: {bm['avg_return_per_trade']:.4f}\n")
                
            if 'tail_metrics' in metrics_test:
                tm = metrics_test['tail_metrics']
                f.write(f"\n#### Big Move Detection Performance\n")
                f.write(f"- Precision (Up): {tm['precision_up_strict']:.2f} (Predicted: {tm['count_pred_up']})\n")
                f.write(f"- Recall (Up): {tm['recall_up_strict']:.2f} (Actual: {tm['count_actual_up']})\n")
                f.write(f"- Precision (Down): {tm['precision_down_strict']:.2f} (Predicted: {tm['count_pred_down']})\n")
                f.write(f"- Recall (Down): {tm['recall_down_strict']:.2f} (Actual: {tm['count_actual_down']})\n")
            
            f.write("\n")
            
            f.write("## Features Used\n")
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write(f"List: {', '.join(feature_cols)}\n")
            
    def save_plot(self, plt_figure, filename="forecast_plot.png"):
        path = os.path.join(self.run_dir, filename)
        plt_figure.savefig(path)
        print(f"Plot saved to {path}")
        
    def plot_scatter(self, y_true, y_pred, title="Predicted vs Actual", filename="scatter_plot.png"):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel("Actual Return")
        plt.ylabel("Predicted Return")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        self.save_plot(fig, filename)
