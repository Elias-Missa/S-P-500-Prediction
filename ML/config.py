# ML Configuration

# Data Paths
DATA_PATH = r'C:\Users\eomis\SP500 Project\S-P-500-Prediction\Output\final_features_v6.csv'

# Target Definition
TARGET_HORIZON = 21  # 1 Month (Trading Days)
TARGET_COL = 'Target_1M'

# Splitting Parameters
TEST_START_DATE = '2023-01-01'
TRAIN_WINDOW_YEARS = 5
VAL_WINDOW_MONTHS = 6
BUFFER_DAYS = 21  # Embargo period to prevent leakage (should be >= TARGET_HORIZON)

# Model Parameters
# Options: 'LinearRegression', 'RandomForest', 'XGBoost', 'MLP'
MODEL_TYPE = 'XGBoost'

# Random Forest Params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# XGBoost Params
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5

# MLP Params
MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER = 500
