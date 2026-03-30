import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent

ML_READY_CSV = current_dir/"data"/"ml_ready.csv"
PLOT_OUTPUT  = current_dir/"data"/"predicted_vs_actual.png"

# Data split parameters
RANDOM_STATE = 42
TEST_SIZE    = 0.2

FEATURE_COLS = ["t2m", "d2m", "tp", "u10", "v10", "swvl1", "wind_speed", "relative_humidity"]
TARGET_COL   = "risk_score"

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(ML_READY_CSV)
    print(f"[INFO] Loaded dataset: {df.shape}")
    print(f"[INFO] Days with fires: {(df[TARGET_COL] > 0).sum()} / {len(df)}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Random 80/20 split, RANDOM_STATE ensures same split every run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"[INFO] Train size: {len(X_train)} rows")
    print(f"[INFO] Test size:  {len(X_test)} rows\n")
    return X_train, X_test, y_train, y_test
