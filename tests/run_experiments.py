import subprocess
import sys
import time

models = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM']

def run_experiments():
    print("Starting Experiment Suite...")
    
    for model in models:
        print(f"\n========================================")
        print(f"Running Experiment: {model}")
        print(f"========================================")
        
        start_time = time.time()
        try:
            # Run the training script with the model type override
            result = subprocess.run(
                [sys.executable, '-m', 'ML.train', '--model_type', model],
                check=True,
                capture_output=False # Let output flow to console
            )
            duration = time.time() - start_time
            print(f"Successfully ran {model} in {duration:.2f} seconds.")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {model}: {e}")
            # Continue to next model even if one fails
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_experiments()
