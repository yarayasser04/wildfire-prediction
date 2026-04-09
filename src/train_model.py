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
    "dist_fire_to_cell",  # Distance (miles) from fire point to this grid cell
                          #   NaN for no-fire rows — filled with 0 below
]

TARGET_COL = "risk_score"


def load_data():
    df = pd.read_csv(ML_READY_CSV)
    print(f"[INFO] Loaded dataset: {df.shape}")

    df["dist_fire_to_cell"] = df["dist_fire_to_cell"].fillna(0.0)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = set(FEATURE_COLS) - set(available)
    if missing:
        print(f"[WARNING] Feature columns not found and skipped: {missing}")

    X = df[available]
    y = (df[TARGET_COL] > 0).astype(int)

    print(f"[INFO] Fire rows:    {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"[INFO] No-fire rows: {(y == 0).sum()} ({(1 - y.mean())*100:.1f}%)")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\n[INFO] Train:      {len(X_train)} rows ({len(X_train)/len(df)*100:.0f}%)")
    print(f"[INFO] Validation: {len(X_val)}   rows ({len(X_val)/len(df)*100:.0f}%)")
    print(f"[INFO] Test:       {len(X_test)}  rows ({len(X_test)/len(df)*100:.0f}%)\n")

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

'''
def plot_results(
    y_val,  y_pred_proba_val,  y_pred_class_val,  cm_val,
    y_test, y_pred_proba_test, y_pred_class_test, cm_test,
    model,  feature_cols: list[str],
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Row 1: Validation ---
    ConfusionMatrixDisplay(cm_val, display_labels=["No Fire", "Fire"]).plot(
        ax=axes[0, 0], colorbar=False, cmap="Blues"
    )
    axes[0, 0].set_title("Validation — Confusion Matrix", fontsize=12)

    RocCurveDisplay.from_predictions(y_val, y_pred_proba_val, ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Validation — ROC Curve", fontsize=12)
    axes[0, 1].plot([0, 1], [0, 1], "r--", linewidth=1)

    axes[0, 2].hist(y_pred_proba_val[y_val == 0], bins=20, alpha=0.6, color="steelblue", label="No Fire")
    axes[0, 2].hist(y_pred_proba_val[y_val == 1], bins=20, alpha=0.6, color="tomato",    label="Fire")
    axes[0, 2].set_xlabel("Predicted Probability of Fire")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Validation — Predicted Probability Distribution", fontsize=12)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # --- Row 2: Test ---
    ConfusionMatrixDisplay(cm_test, display_labels=["No Fire", "Fire"]).plot(
        ax=axes[1, 0], colorbar=False, cmap="Oranges"
    )
    axes[1, 0].set_title("Test — Confusion Matrix", fontsize=12)

    RocCurveDisplay.from_predictions(y_test, y_pred_proba_test, ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("Test — ROC Curve", fontsize=12)
    axes[1, 1].plot([0, 1], [0, 1], "r--", linewidth=1)

    # Feature importance — uses actual feature names after dropping missing columns
    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    axes[1, 2].barh(feat_imp.index, feat_imp.values, color="mediumpurple")
    axes[1, 2].set_title("Feature Importances", fontsize=12)
    axes[1, 2].set_xlabel("Importance Score")
    axes[1, 2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("XGBoost Wildfire Risk — Classification Results", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {PLOT_OUTPUT}")
    plt.show()
'''

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
    model = train_model(X_train, y_train, X_val, y_val)

    y_pred_proba_val,  y_pred_class_val,  auc_val,  cm_val  = evaluate_split(model, X_val,  y_val,  "VALIDATION")
    y_pred_proba_test, y_pred_class_test, auc_test, cm_test = evaluate_split(model, X_test, y_test, "TEST")
'''
    plot_results(
        y_val,  y_pred_proba_val,  y_pred_class_val,  cm_val,
        y_test, y_pred_proba_test, y_pred_class_test, cm_test,
        model,  feature_cols,
    )'''