import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import List

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent

WEATHER_DATA_DIR  = current_dir / "data" / "weather_data"
WILDFIRE_DATA_DIR = current_dir / "data" / "wildfire_data" / "AK_fire_location_points_NAD83.csv"
OUTPUT_CSV        = current_dir / "data" / "ml_ready.csv"

YEARS  = range(1950, 1991)
MONTHS = ["05", "06", "07", "08"]

FAIRBANKS_LAT = 64.8378
FAIRBANKS_LON = -147.7164
RADIUS_MILES  = 50.0

# Risk weight per cause — higher = riskier
# Lightning = highest risk (natural, unpredictable, hardest to control)
# Human/Structure = moderate (preventable but still dangerous)
# Hand pile/slash = lower (controlled burns, usually managed)
# Undetermined = low default since cause is unknown

CAUSE_FILTER = "lightning"

CAUSE_WEIGHTS = {
    "lightning":                              1.0,
    "electrical transmission/distribution":   0.8,
    "railroad":                               0.7,
    "military ordnance":                      0.7,
    "incendiary":                             0.7,
    "human":                                  0.7,
    "smoking":                                0.6,
    "campfire":                               0.6,
    "recreation":                             0.6,
    "structure":                              0.6,
    "passenger vehicle/motorized rv":         0.5,
    "miscellaneous":                          0.5,
    "debris burning":                         0.4,
    "debris bng":                             0.4,
    "hand pile/slash":                        0.3,
    "pile burning":                           0.3,
    "not investigated":                       0.2,
    "undetermined":                           0.2,
}
DEFAULT_WEIGHT = 0.2  # used for any other cause not listed above

# Haversine — vectorized so it works on both scalars and numpy arrays
def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in miles.
    lat1/lon1 can be scalar (fire point).
    lat2/lon2 can be arrays (weather grid).
    """
    R = 3958.8
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# Weather loading
def get_files(data_dir: str) -> List[str]:
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
    
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine="netcdf4",
        chunks="auto",
        join="override",
    )
    
    if "number" in ds.coords:
        ds = ds.drop_vars("number")
        
    ds = ds.sortby(["latitude", "longitude", "valid_time"])

    lats = ds.latitude.values
    lons = ds.longitude.values

    mask = np.array([
        [haversine_miles(FAIRBANKS_LAT, FAIRBANKS_LON, la, lo) <= RADIUS_MILES
         for lo in lons]
        for la in lats
    ], dtype=bool)

    n_cells = mask.sum()
    print(f"[INFO] Grid cells within {RADIUS_MILES} miles of Fairbanks: {n_cells}")
    if n_cells == 0:
        raise ValueError(
            "No weather grid cells found within the Fairbanks radius. "
            "Check FAIRBANKS_LAT/LON and the grid coverage of your .nc files."
        )

    mask_da = xr.DataArray(mask, dims=["latitude", "longitude"])
    ds_filtered = ds.where(mask_da)

    ds_filtered["t2m"] = (ds_filtered["t2m"] - 273.15) * 9 / 5 + 32
    ds_filtered["d2m"] = (ds_filtered["d2m"] - 273.15) * 9 / 5 + 32
    ds_mean = ds_filtered.mean(dim=["latitude", "longitude"])


    df = ds_mean.to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.groupby("date").mean(numeric_only=True).reset_index()

    # Engineer wind speed: sqrt(u² + v²)
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    # Engineer relative humidity via Magnus approximation
    # t2m and d2m are now reliably in Fahrenheit — convert to Celsius for formula
    if "t2m" in df.columns and "d2m" in df.columns:
        t_c = (df["t2m"] - 32) * 5 / 9
        d_c = (df["d2m"] - 32) * 5 / 9
        df["relative_humidity"] = (
            100
            * np.exp((17.625 * d_c) / (243.04 + d_c))
            / np.exp((17.625 * t_c) / (243.04 + t_c))
        )

    print(f"[INFO] Weather DataFrame shape: {df.shape}")
    return df


# Wildfire loading
def load_wildfire(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(
        df["DISCOVERYDATETIME"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    df["date"] = df["date"].apply(
        lambda x: x.replace(year=x.year - 100)
        if pd.notnull(x) and x.year > 2000 else x
    )
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date"])
    df = df[df["FIRESEASON"].astype(str).str.strip().isin([str(y) for y in YEARS])]
    df = df[df["date"].dt.month.isin([int(m) for m in MONTHS])]
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    before = len(df)
    dist_to_fairbanks = haversine_miles(
        FAIRBANKS_LAT, FAIRBANKS_LON,
        df["LATITUDE"].values,
        df["LONGITUDE"].values,
    )
    df = df[dist_to_fairbanks <= RADIUS_MILES].copy()
    print(
        f"[INFO] Wildfire records: {before} in study window, "
        f"{len(df)} within {RADIUS_MILES} miles of Fairbanks"
    )

    df["cause_lower"] = df["SPECIFICCAUSE"].str.lower().str.strip().fillna("undetermined")
    
    if CAUSE_FILTER is not None:
        before_cause = len(df)
        df = df[df["cause_lower"].str.contains(CAUSE_FILTER, na=False)]
        print(
            f"[INFO] Cause filter '{CAUSE_FILTER}': {before_cause} records → {len(df)} kept"
        )
    df["weight"] = df["cause_lower"].map(
        lambda c: next((v for k, v in CAUSE_WEIGHTS.items() if k in str(c)), DEFAULT_WEIGHT)
    )
    return df[["date", "LATITUDE", "LONGITUDE", "weight"]]

def spatial_join(weather_df: pd.DataFrame, fire_df: pd.DataFrame) -> pd.DataFrame:
    weather_by_date = weather_df.set_index("date")

    results = []
    for date, fires_on_date in fire_df.groupby("date"):
        if date not in weather_by_date.index:
            continue

        weather_row = weather_by_date.loc[date]

        if isinstance(weather_row, pd.DataFrame):
            weather_row = weather_row.mean(numeric_only=True)

        for _, fire in fires_on_date.iterrows():
            row = weather_row.copy()
            row["date"]        = date
            row["fire_lat"]    = fire["LATITUDE"] 
            row["fire_lon"]    = fire["LONGITUDE"] 
            row["fire_weight"] = fire["weight"]
            results.append(row)

    return pd.DataFrame(results)

def add_no_fire_days(weather_df: pd.DataFrame, fire_df: pd.DataFrame) -> pd.DataFrame:
    fire_dates = set(fire_df["date"])

    no_fire = weather_df[~weather_df["date"].isin(fire_dates)].copy()
    n_sample = min(len(fire_dates), len(no_fire))
    no_fire  = no_fire.sample(n=n_sample, random_state=42)

    no_fire["fire_lat"]    = np.nan
    no_fire["fire_lon"]    = np.nan
    no_fire["fire_weight"] = 0.0
    no_fire["risk_score"]  = 0.0

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
    no_fire_data = add_no_fire_days(weather, fire)

    if "fire_weight" in fire_data.columns:
        fire_data["risk_score"] = fire_data["fire_weight"]

    df = pd.concat([fire_data, no_fire_data], ignore_index=True)
    df = df.dropna(subset=[c for c in df.columns if c not in ("fire_lat", "fire_lon")])

    max_score = df["risk_score"].max()
    if max_score > 0:
        df["risk_score"] = df["risk_score"] / max_score

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Saved ML-ready dataset to: {OUTPUT_CSV}")
    print(f"[INFO] Final shape: {df.shape}")
    print(f"[INFO] Fire rows: {(df['risk_score'] > 0).sum()}  |  No-fire rows: {(df['risk_score'] == 0).sum()}")
    print(f"\n{df.head(20)}")
    return df
if __name__ == "__main__":
    build_dataset()