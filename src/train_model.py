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

def load_data():
    df = pd.read_csv(ML_READY_CSV)
    print(f"[INFO] Loaded dataset: {df.shape}")
    print(f"[INFO] Days with fires: {(df[TARGET_COL] > 0).sum()} / {len(df)}")
 
    X = df[FEATURE_COLS]
    y_raw = df[TARGET_COL]
    y = np.log1p(y_raw)
 

    # Random 80/20 split, RANDOM_STATE ensures same split every run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    _, _, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"[INFO] Train size: {len(X_train)} rows")
    print(f"[INFO] Test size:  {len(X_test)} rows\n")
    return X_train, X_test, y_train, y_test, y_test_raw

def train_model(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete\n")
    return model

def evaluate_model(model, X_test, y_test, y_test_raw):
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_pred     = np.clip(y_pred, 0, 1)
 
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2   = r2_score(y_test_raw, y_pred)
 
    print(f"[RESULTS] RMSE : {rmse:.4f}")
    print(f"[RESULTS] R²   : {r2:.4f}\n")
    return y_pred

def plot_predictions(y_test_raw, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ax.scatter(y_test_raw, y_pred, alpha=0.4, edgecolors="k",
               linewidths=0.3, color="steelblue", label="Predictions")
 
    min_val = min(y_test_raw.min(), y_pred.min())
    max_val = max(y_test_raw.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
 
    ax.set_xlabel("Actual Risk Score", fontsize=12)
    ax.set_ylabel("Predicted Risk Score", fontsize=12)
    ax.set_title("XGBoost: Predicted vs Actual Wildfire Risk Score", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"[INFO] Prediction plot saved to: {PLOT_OUTPUT}")
    plt.show()
 
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, y_test_raw = load_data()
    model  = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test, y_test_raw)
    plot_predictions(y_test_raw, y_pred)
 