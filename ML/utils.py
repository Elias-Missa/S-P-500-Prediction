import os
import datetime
import shutil
from . import config

class ExperimentLogger:
    def __init__(self, model_name="Model", process_tag="Static"):
        # Create base ML_Output directory
        self.base_dir = r'C:\Users\eomis\SP500 Project\S-P-500-Prediction\ML_Output'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
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
        Writes a markdown summary of the run.
        """
        summary_path = os.path.join(self.run_dir, "summary.md")
        
        with open(summary_path, 'w') as f:
            f.write(f"# ML Run Summary\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- **Model**: `{model_type}`\n")
            f.write(f"- **Target Horizon**: {config.TARGET_HORIZON} days\n")
            f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
            f.write(f"- **Val Window**: {config.VAL_WINDOW_MONTHS} months\n")
            f.write(f"- **Buffer**: {config.BUFFER_DAYS} days\n")
            f.write(f"- **Test Start**: {config.TEST_START_DATE}\n\n")
            
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
            
            f.write("## Metrics\n")
            f.write("### Validation\n")
            f.write(f"- RMSE: {metrics_val['rmse']:.6f}\n")
            f.write(f"- MAE: {metrics_val['mae']:.6f}\n")
            f.write(f"- Directional Accuracy: {metrics_val['dir_acc']:.2f}%\n\n")
            
            f.write("### Test (Out of Sample)\n")
            f.write(f"- RMSE: {metrics_test['rmse']:.6f}\n")
            f.write(f"- MAE: {metrics_test['mae']:.6f}\n")
            f.write(f"- Directional Accuracy: {metrics_test['dir_acc']:.2f}%\n\n")
            
            f.write("## Features Used\n")
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write(f"List: {', '.join(feature_cols)}\n")
            
    def save_plot(self, plt_figure, filename="forecast_plot.png"):
        path = os.path.join(self.run_dir, filename)
        plt_figure.savefig(path)
        print(f"Plot saved to {path}")
