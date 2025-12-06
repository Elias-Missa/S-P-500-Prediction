# ML Configuration

# Data Paths
DATA_PATH = r'C:\Users\eomis\SP500 Project\S-P-500-Prediction\Output\final_features_v6.csv'

# Target Definition
TARGET_HORIZON = 21  # 1 Month (Trading Days)
TARGET_COL = 'Target_1M'

# Splitting Parameters
TEST_START_DATE = '2023-01-01'
TRAIN_START_DATE = '2010-01-01' # If set, overrides TRAIN_WINDOW_YEARS for an Expanding Window
TRAIN_WINDOW_YEARS = None # Set to integer (e.g., 5) for Rolling Window, or None to use TRAIN_START_DATE
VAL_WINDOW_MONTHS = 6
BUFFER_DAYS = 21  # Embargo period to prevent leakage (should be >= TARGET_HORIZON)

# Model Parameters
# Options: 'LinearRegression', 'RandomForest', 'XGBoost', 'MLP'
MODEL_TYPE = 'LSTM'

# Random Forest Params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_RANDOM_STATE = 42

# XGBoost Params
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5

# MLP Params
MLP_HIDDEN_LAYERS = (64, 32)
MLP_LEARNING_RATE_INIT = 0.001
MLP_ALPHA = 0.0001
MLP_MAX_ITER = 500

# LSTM Params
LSTM_TIME_STEPS = 10
LSTM_HIDDEN_DIM = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# Optuna Tuning
USE_OPTUNA = True
OPTUNA_TRIALS = 20
