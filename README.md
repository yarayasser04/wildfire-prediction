# Wildfire Risk Prediction — Fairbanks, Alaska

A machine learning pipeline that predicts lightning-caused wildfire risk in the Fairbanks region. It trains an XGBoost classifier on 11 years of ERA5 weather and Alaska fire data, then forecasts fire risk for any future year using linear regression–based weather projections.

**Authors:** Capstone Project — University of Alaska Fairbanks

---

## Project Structure

```
capstone_wildfire_prediction/
├── src/
│   ├── ml_model.py            # Loads weather & fire data, spatial join → ml_ready.csv
│   ├── train_model.py         # XGBoost classifier: train, evaluate, predict
│   ├── train_model_randf.py   # RandomForest alternative with GridSearchCV
│   └── weather_predict.py     # Linear regression weather forecasting
├── tools/
│   └── file_checker.py        # Validates and extracts NetCDF files from ZIPs
├── notebooks/
│   ├── wildfire_map.ipynb     # Geographic heatmap visualizations
│   └── wildfire_graphs.ipynb  # Statistical plots
├── data/                      # Not tracked in git — see Data section
└── requirements.txt
```

---

## Data

The `data/` directory is not tracked in git. You need:

| File/Folder | Description |
|---|---|
| `data/weather_data/` | NetCDF4 files named `fairbanks_YYYY_MM_real.nc` (2000–2010, May–Aug) |
| `data/wildfire_data/AK_fire_location_points_NAD83.csv` | Alaska fire location points |

**Weather variables:** 2 m temperature (`t2m`), dew point (`d2m`), total precipitation (`tp`), 10 m wind components (`u10`, `v10`), volumetric soil water layer 1 (`swvl1`).

---

## Setup

**Requirements:** Python 3.11

```bash
pip install -r requirements.txt

# Extract any nested NetCDF files from ZIP archives
python tools/file_checker.py

# Build the ML-ready dataset
python src/ml_model.py

# Train XGBoost, evaluate, and predict future fire risk
python src/train_model.py

# (Optional) Train RandomForest regressor
python src/train_model_randf.py
```

To predict a different future year, update `FUTURE_YEAR` in `src/train_model.py`.  
To change the forecast month, update `MONTH_PREDICT` in `src/weather_predict.py`.

---

## Pipeline

```
NetCDF weather files ──► load_weather()  ─┐
                                           ├─► spatial_join() ──► ml_ready.csv
Alaska fire CSV ──────► load_wildfire() ──┘

ml_ready.csv ─────────► train_model.py ──────────────► XGBoost ──► scored predictions
              └──────► weather_predict.py ──► future weather ──► future fire risk
```

1. **`ml_model.py`** — Aligns weather and fire data. For each fire event, finds all weather grid cells within 100 miles (Haversine) on the same date. Samples no-fire days at a 2:1 ratio. Outputs `ml_ready.csv`.

2. **`train_model.py`** — Trains an XGBoost binary classifier (fire/no-fire) with a chronological 70/15/15 split. Calls `weather_predict.py` to generate future weather, then produces per-grid fire probability predictions.

3. **`weather_predict.py`** — Fits per-grid linear regressions on historical monthly weather to extrapolate values for a target year. Called automatically by `train_model.py` but can also run standalone.

4. **`train_model_randf.py`** — Alternative RandomForest regressor with GridSearchCV tuning on the same dataset.

---

## Models

### XGBoost Classifier

| Setting | Value |
|---|---|
| Features | t2m, d2m, tp, u10, v10, swvl1, wind_speed, relative_humidity, grid_lat, grid_lon |
| Target | Binary: fire vs. no-fire |
| n_estimators | 500 (early stopping on val AUC) |
| max_depth | 4 |
| learning_rate | 0.05 |
| scale_pos_weight | Computed from class ratio |
| Data split | Chronological 70 / 15 / 15 |

Derived features: `wind_speed = sqrt(u10² + v10²)` and `relative_humidity` via the Magnus approximation.

### RandomForest Regressor

Predicts a continuous `risk_score` using the same 8 weather features (no spatial coordinates). GridSearchCV tunes `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

---

## Outputs

| File | Description |
|---|---|
| `data/ml_ready.csv` | Full processed dataset (~2.8 M rows) |
| `data/ml_ready_scored_grid.csv` | Per-grid fire probability for historical dates |
| `data/ml_ready_scored.csv` | Daily max fire probability across all grid cells |
| `data/future_fire_risk_YYYY.csv` | Per-grid fire probability for predicted future year |
| `data/weather_predictions_YYYY.csv` | Projected weather features for future year |
| `data/classification_results.png` | ROC curve and confusion matrix |

---

## Key Parameters

| Parameter | Default | File |
|---|---|---|
| Study years | 2000–2010 | `ml_model.py` → `YEARS` |
| Fire season months | May–Aug | `ml_model.py` → `MONTHS` |
| Cause filter | `"lightning"` | `ml_model.py` → `CAUSE_FILTER` |
| Spatial join radius | 100 miles | `ml_model.py` → `SPATIAL_JOIN_RADIUS_MILES` |
| No-fire sampling ratio | 2:1 | `ml_model.py` → `add_no_fire_days(ratio=2)` |
| Future prediction year | 2025 | `train_model.py` → `FUTURE_YEAR` |
| Forecast month | June | `weather_predict.py` → `MONTH_PREDICT` |
