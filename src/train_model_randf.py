import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent
ML_READY_CSV = current_dir/"data"/"ml_ready.csv"
PLOT_OUTPUT  = current_dir/"data"/"predicted_vs_actual_randf.png"

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



def train_model(X_train, y_train) -> RandomForestRegressor:

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf= 4,
        min_samples_split= 10,
        max_depth=5, 
        random_state=RANDOM_STATE, 
        n_jobs=-1)

    model.fit(X_train, y_train)
    return(model)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[INFO] RF MSE: {mse:.4f}")
    print(f"[INFO] RF R2:  {r2:.4f}")
    return(y_pred)

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (Random Forest)")
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved to: {PLOT_OUTPUT}")



def optimize_param(X_train, y_train):  
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Use a new GridSearchCV object
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        cv=3,          # cross-validation
        scoring='neg_mean_squared_error',
        verbose=1
    )
    X_train, _, y_train, _ = load_data()
    grid_search.fit(X_train, y_train)
    print(f"[INFO] Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return(best_model)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model  = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, y_pred)
    optimize_param(X_train, y_train)