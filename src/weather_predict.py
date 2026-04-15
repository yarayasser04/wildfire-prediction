import pandas as pd
from pathlib import Path
from ml_model import haversine_miles
from sklearn.linear_model import LinearRegression

current_file_path = Path(__file__).resolve()
current_dir       = current_file_path.parent.parent
WEATHER_DATA_DIR  = current_dir / "data" / "weather_data"
ML_READY          = current_dir / "data" / "ml_ready.csv"

FAIRBANKS_LAT = 64.8378
FAIRBANKS_LON = -147.7164
SPATIAL_JOIN_RADIUS_MILES = 100.0

YEAR_PREDICT = 2009
MONTH_PREDICT = "06"

# Find all columns
try:
    if ML_READY.exists():
        _sample = pd.read_csv(ML_READY, nrows=10)
        _exclude = {"date", "grid_lat", "grid_lon", "year", "month"}
        PREDICT_COLUMNS = [
            c for c in _sample.select_dtypes(include=["number"]).columns if c not in _exclude
        ]
    else:
        PREDICT_COLUMNS = ["t2m"]
except Exception:
    PREDICT_COLUMNS = ["t2m"]

def predict_june_regression(
    weather_df: pd.DataFrame,
    predict_year: int,
    center_lat: float = FAIRBANKS_LAT,
    center_lon: float = FAIRBANKS_LON,
    radius_miles: float = SPATIAL_JOIN_RADIUS_MILES,
) -> pd.DataFrame:
    df = weather_df.copy()
    
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month.astype(str).str.zfill(2)
    df["year"] = df["date"].dt.year
    
    df = df[df["month"] == MONTH_PREDICT]
    if df.empty: return pd.DataFrame()

    grids = df[["grid_lat", "grid_lon"]].drop_duplicates().reset_index(drop=True)
    grids["dist_miles"] = haversine_miles(grids["grid_lat"], grids["grid_lon"], center_lat, center_lon)
    nearby = grids[grids["dist_miles"] <= radius_miles][["grid_lat", "grid_lon"]]
    
    if nearby.empty: return pd.DataFrame()
    df = df.merge(nearby, on=["grid_lat", "grid_lon"], how="inner")

    targets = [t for t in PREDICT_COLUMNS if t in df.columns]    
    all_preds = []

    for target in targets:
        print(f"Predicting target: {target}")
        
        # Calculate mean
        monthly = df.groupby(["grid_lat", "grid_lon", "year"])[target].mean().reset_index()
        
        #Drop NAN
        monthly = monthly.dropna(subset=[target])

        for (glat, glon), grp in monthly.groupby(["grid_lat", "grid_lon"]):
            years = grp["year"].values.reshape(-1, 1)
            vals = grp[target].values
            
            if len(years) == 0:
                continue 
            
            if len(years) < 2:
                pred_val = float(vals[-1])
            else:
                model = LinearRegression()
                model.fit(years, vals)

                pred_val = float(model.predict([[predict_year]])[0])
            
            all_preds.append({
                "grid_lat": glat, 
                "grid_lon": glon, 
                "target": target, 
                "pred_val": pred_val
            })

    if not all_preds: return pd.DataFrame()

    # Pivot the results so each target is its own column
    res_df = pd.DataFrame(all_preds)
    res_df = res_df.pivot_table(
        index=["grid_lat", "grid_lon"], 
        columns="target", 
        values="pred_val"
    ).reset_index()
    
    # Clean up column names (remove "pred_val" hierarchy)
    res_df.columns.name = None
    res_df["pred_year"] = predict_year
    res_df["month"] = MONTH_PREDICT
    
    return res_df

if __name__ == "__main__":
    weather = pd.read_csv(ML_READY)
    
    print(f"Columns identified for prediction: {PREDICT_COLUMNS}")
    
    monthly_preds = predict_june_regression(
        weather, 
        YEAR_PREDICT, 
        FAIRBANKS_LAT, 
        FAIRBANKS_LON, 
        SPATIAL_JOIN_RADIUS_MILES
    )

    out_path = ML_READY.parent / f"weather_predictions_{YEAR_PREDICT}.csv"
    monthly_preds.to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions to: {out_path}")
