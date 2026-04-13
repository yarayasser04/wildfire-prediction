import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir       = current_file_path.parent.parent

WEATHER_DATA_DIR  = current_dir / "data" / "weather_data"
WILDFIRE_DATA_DIR = current_dir / "data" / "wildfire_data" / "AK_fire_location_points_NAD83.csv"
OUTPUT_CSV        = current_dir / "data" / "ml_ready.csv"

YEARS  = range(2000, 2008)
MONTHS = ["05", "06", "07", "08"]

FAIRBANKS_LAT = 64.8378
FAIRBANKS_LON = -147.7164

SPATIAL_JOIN_RADIUS_MILES = 100.0

CAUSE_FILTER = "lightning"

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))



# Weather loading — keeps one row per (date, grid_lat, grid_lon)
def get_files(data_dir: str) -> list[str]:
    files = []
    for year in YEARS:
        for month in MONTHS:
            pattern = os.path.join(data_dir, f"fairbanks_{year}_{month}_real.nc")
            matched = sorted(glob.glob(pattern))
            if not matched:
                print(f"[WARNING] No files found for {year}-{month}")
            files.extend(matched)
    return files


def load_weather(data_dir: str) -> pd.DataFrame:
    files = get_files(data_dir)
    print(f"[INFO] Found {len(files)} weather files")

    datasets = []
    for f in files:
        ds_single = xr.open_dataset(f, engine="netcdf4")
        if "number" in ds_single.coords:
            ds_single = ds_single.drop_vars("number")
        datasets.append(ds_single)
    ds = xr.concat(datasets, dim="valid_time")

    ds = ds.sortby(["latitude", "longitude", "valid_time"])

    # Kelvin → Fahrenheit
    ds["t2m"] = (ds["t2m"] - 273.15) * 9 / 5 + 32
    ds["d2m"] = (ds["d2m"] - 273.15) * 9 / 5 + 32

    df = ds.to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "date", "latitude": "grid_lat", "longitude": "grid_lon"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    group_cols = ["date", "grid_lat", "grid_lon"]
    df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
    df = df.dropna(subset=["t2m"])

    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    if "t2m" in df.columns and "d2m" in df.columns:
        t_c = (df["t2m"] - 32) * 5 / 9
        d_c = (df["d2m"] - 32) * 5 / 9
        df["relative_humidity"] = (
            100
            * np.exp((17.625 * d_c) / (243.04 + d_c))
            / np.exp((17.625 * t_c) / (243.04 + t_c))
        )
        
    return df



# Wildfire loading - keeps one row per fire event, with weight=1.0 for all lightning-caused fires.
# Spatial filtering to nearby grid cells happens in spatial_join.
def load_wildfire(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    df["date"] = pd.to_datetime(
        df["DISCOVERYDATETIME"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    
    df["date"] = df["date"].apply(
        lambda x: x.replace(year=x.year - 100)
        if pd.notnull(x) and x.year > 2068
        else x
    )
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date"])

    # Keep only study years and months
    df = df[df["FIRESEASON"].astype(str).str.strip().isin([str(y) for y in YEARS])]
    df = df[df["date"].dt.month.isin([int(m) for m in MONTHS])]
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    print(f"[INFO] Wildfire records in study window: {len(df)}")

    df["cause_lower"] = (
        df["SPECIFICCAUSE"].str.lower().str.strip().fillna("undetermined")
    )
    before_cause = len(df)
    df = df[df["cause_lower"].str.contains(CAUSE_FILTER, na=False)].copy()
    print(
        f"[INFO] Cause filter '{CAUSE_FILTER}': {before_cause} records → {len(df)} kept"
    )
    
    df["weight"] = 1.0

    return df[["date", "LATITUDE", "LONGITUDE", "weight"]]



# Spatial join — fire point × nearby weather grid cells
# For every fire event, find all weather grid cells within SPATIAL_JOIN_RADIUS_MILES on the same date and produce one combined row per (fire, grid_cell) pair
def spatial_join(weather_df: pd.DataFrame, fire_df: pd.DataFrame) -> pd.DataFrame:
    weather_by_date = {
        date: grp.reset_index(drop=True)
        for date, grp in weather_df.groupby("date")
    }
    

    results = []
    for _, fire in fire_df.iterrows():
        date = fire["date"]
        if date not in weather_by_date:
            continue

        wx = weather_by_date[date]

        dists = haversine_miles(
            fire["LATITUDE"], fire["LONGITUDE"],
            wx["grid_lat"].values,
            wx["grid_lon"].values,
        )
        nearby = wx[dists <= SPATIAL_JOIN_RADIUS_MILES].copy()
        if nearby.empty:
            continue

        nearby["fire_lat"]         = fire["LATITUDE"]
        nearby["fire_lon"]         = fire["LONGITUDE"]
        nearby["fire_weight"]      = fire["weight"]
        nearby["dist_fire_to_cell"] = dists[dists <= SPATIAL_JOIN_RADIUS_MILES]
        results.append(nearby)

    if not results:
        raise ValueError("spatial_join produced no rows — check radii and date overlap.")

    out = pd.concat(results, ignore_index=True)
    out["date"] = out["date"]  
    print(f"[INFO] Fire rows after spatial join: {len(out)}")
    return out

def add_no_fire_days(weather_df, fire_df, fire_row_count: int, ratio: int = 2) -> pd.DataFrame:
    fire_dates   = set(fire_df["date"])
    no_fire_pool = weather_df[~weather_df["date"].isin(fire_dates)].copy()

    unique_dates = no_fire_pool["date"].drop_duplicates()
    n_rows_per_date = len(no_fire_pool) / len(unique_dates) if len(unique_dates) > 0 else 1
    n_days = min(int(fire_row_count * ratio / n_rows_per_date) + 1, len(unique_dates))
    no_fire_dates = unique_dates.sample(n=n_days, random_state=42)
    no_fire = no_fire_pool[no_fire_pool["date"].isin(no_fire_dates)]
 
    no_fire["fire_lat"]          = np.nan
    no_fire["fire_lon"]          = np.nan
    no_fire["fire_weight"]       = 0.0
    no_fire["dist_fire_to_cell"] = np.nan
    no_fire["risk_score"]        = 0.0
 
    print(f"[INFO] No-fire rows sampled ({ratio}:1 ratio): {len(no_fire)}")
    return no_fire


# Build dataset
def build_dataset() -> pd.DataFrame:
    if not os.path.exists(WEATHER_DATA_DIR):
        raise FileNotFoundError(f"Weather data directory not found: {WEATHER_DATA_DIR}")
    if not os.path.exists(WILDFIRE_DATA_DIR):
        raise FileNotFoundError(f"Wildfire CSV not found: {WILDFIRE_DATA_DIR}")

    weather = load_weather(WEATHER_DATA_DIR)
    fire    = load_wildfire(WILDFIRE_DATA_DIR)

    fire_data    = spatial_join(weather, fire)
    no_fire_data = add_no_fire_days(weather, fire, fire_row_count=len(fire_data), ratio=2)

    fire_data["risk_score"] = 1.0

    df = pd.concat([fire_data, no_fire_data], ignore_index=True)

    non_spatial = [c for c in df.columns if c not in ("fire_lat", "fire_lon", "dist_fire_to_cell")]
    df = df.dropna(subset=non_spatial)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Saved ML-ready dataset to: {OUTPUT_CSV}")
    print(f"[INFO] Final shape: {df.shape}")
    print(f"[INFO] Fire rows:    {(df['risk_score'] > 0).sum()}")
    print(f"[INFO] No-fire rows: {(df['risk_score'] == 0).sum()}")
    print(f"\n{df.head(20)}")
    return df


if __name__ == "__main__":
    build_dataset()