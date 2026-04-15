import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay,
)
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir       = current_file_path.parent.parent

ML_READY_CSV = current_dir / "data" / "ml_ready.csv"
PLOT_OUTPUT  = current_dir / "data" / "classification_results.png"

RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15

FEATURE_COLS = [
    "t2m",                # Air temperature at 2 m — hotter → higher fire risk
    "d2m",                # Dew point at 2 m — lower → drier air → higher fire risk
    "tp",                 # Total precipitation — more rain → lower fire risk
    "u10",                # Eastward wind at 10 m
    "v10",                # Northward wind at 10 m
    "swvl1",              # Top-layer soil moisture — drier soil → more fuel
    "wind_speed",         # Derived: sqrt(u10² + v10²)
    "relative_humidity",  # Derived via Magnus approximation
    "grid_lat",           # Grid cell latitude  — spatial context for the model
    "grid_lon",           # Grid cell longitude — spatial context for the model

]

TARGET_COL = "risk_score"


def load_data():
    df = pd.read_csv(ML_READY_CSV)
    
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = (df[TARGET_COL] > 0).astype(int)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[available]
    y = (df_sorted[TARGET_COL] > 0).astype(int)

    n = len(df_sorted)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end],      y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],         y.iloc[val_end:]

    print(f"[INFO] Train dates: {df_sorted['date'].iloc[0]} → {df_sorted['date'].iloc[train_end - 1]}")
    print(f"[INFO] Val dates:   {df_sorted['date'].iloc[train_end]} → {df_sorted['date'].iloc[val_end - 1]}")
    print(f"[INFO] Test dates:  {df_sorted['date'].iloc[val_end]} → {df_sorted['date'].iloc[-1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, available


def train_model(X_train, y_train, X_val, y_val) -> XGBClassifier:
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    scale = neg / pos

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=RANDOM_STATE,
        verbosity=0,
        early_stopping_rounds=30,
        eval_metric="auc",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    print(f"\n[INFO] Best iteration: {model.best_iteration}")
    print("[INFO] Model training complete\n")
    return model


def evaluate_split(model, X, y, split_name: str):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred_class = model.predict(X)
    auc    = roc_auc_score(y, y_pred_proba)
    cm     = confusion_matrix(y, y_pred_class)
    report = classification_report(y, y_pred_class, target_names=["No Fire", "Fire"])

    print(f"{'='*45}")
    print(f"  {split_name} Results")
    print(f"{'='*45}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"\n{report}")
    return y_pred_proba, y_pred_class, auc, cm

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
    model = train_model(X_train, y_train, X_val, y_val)

    y_pred_proba_val,  y_pred_class_val,  auc_val,  cm_val  = evaluate_split(model, X_val,  y_val,  "VALIDATION")
    y_pred_proba_test, y_pred_class_test, auc_test, cm_test = evaluate_split(model, X_test, y_test, "TEST")
    
    df = pd.read_csv(ML_READY_CSV)
    available = [c for c in FEATURE_COLS if c in df.columns]

    df["fire_probability"] = model.predict_proba(df[available])[:, 1]

    GRID_CSV = current_dir / "data" / "ml_ready_scored_grid.csv"
    grid_cols = ["date", "grid_lat", "grid_lon", "fire_probability", "risk_score"]
    df[grid_cols].sort_values(["date", "grid_lat", "grid_lon"]).to_csv(GRID_CSV, index=False)
    print(f"[INFO] Per-grid-cell predictions saved to: {GRID_CSV}")

    daily = (
        df.groupby("date")
        .agg(
            fire_probability=("fire_probability", "max"),
            fire_occurred=("risk_score", lambda s: int((s > 0).any())),
        )
        .reset_index()
        .sort_values("date")
    )

    SCORED_CSV = current_dir / "data" / "ml_ready_scored.csv"
    daily.to_csv(SCORED_CSV, index=False)
    print(f"\n[INFO] Daily scored dataset saved to: {SCORED_CSV}")
    print(daily.to_string())
