import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import List

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent
WEATHER_DATA_DIR = current_dir /"data"/"weather_data"
WILDFIRE_DATA_DIR = current_dir /"data"/"wildfire_data"/"AK_fire_location_points_NAD83.csv"
OUTPUT_CSV = current_dir /"data"/"ml_ready.csv"


YEARS = range(1950, 2025)
MONTHS = ["05", "06", "07", "08"]

# Risk weight per cause — higher = riskier
# Lightning = highest risk (natural, unpredictable, hardest to control)
# Human/Structure = moderate (preventable but still dangerous)
# Hand pile/slash = lower (controlled burns, usually managed)
# Undetermined = low default since cause is unknown
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
 
# Load weather data files based on specified years and months, ensuring they exist
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
    
    files_old = [f for f in files if any(f"_{y}_" in f for y in range(1950, 2018))]
    files_new = [f for f in files if any(f"_{y}_" in f for y in range(2018, 2025))]
    print(f"[INFO] Old grid files: {len(files_old)}, New grid files: {len(files_new)}")
    
    ds_old = xr.open_mfdataset(files_old, combine="by_coords", engine="netcdf4", chunks="auto", join="override")
    
    if files_new:
        ds_new = xr.open_mfdataset(files_new, combine="by_coords", engine="netcdf4", chunks="auto", join="override")
        # Regrid new files onto the old grid
        ds_new = ds_new.interp(latitude=ds_old.latitude, longitude=ds_old.longitude)
        ds = xr.concat([ds_old, ds_new], dim="valid_time")
    else:
        ds = ds_old

    if "number" in ds.coords:
        ds = ds.drop_vars("number")
    
    #Convert temperature from Kelvin to Fahrenheit
    ds["t2m"] = (ds["t2m"] - 273.15) * 9 / 5 + 32
    ds["d2m"] = (ds["d2m"] - 273.15) * 9 / 5 + 32
    
    #Spatially average across all lat/lon grid points → one value per timestep
    ds_mean = ds.mean(dim=["latitude", "longitude"])
    
    #Convert to DataFrame and resample to daily frequency (mean of all timesteps in a day)
    df = ds_mean.to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.groupby("date").mean(numeric_only=True).reset_index()
    
    # Engineer wind speed from u and v components
    # u = east/west, v = north/south — combine into a single speed value
    # Wind speed = sqrt(u^2 + v^2)
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
 
    # Engineer relative humidity from temperature and dew point
    # Using the Magnus approximation formula
    if "t2m" in df.columns and "d2m" in df.columns:
        # Convert F back to C for the formula
        t_c = (df["t2m"] - 32) * 5 / 9
        d_c = (df["d2m"] - 32) * 5 / 9
        df["relative_humidity"] = 100 * np.exp((17.625 * d_c) / (243.04 + d_c)) / \
                                            np.exp((17.625 * t_c) / (243.04 + t_c))
 
    print(f"[INFO] Weather DataFrame shape: {df.shape}")
    return df

# Load wildfire data and engineer a risk score based on cause
def load_risk_scores(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    
    df["date"] = pd.to_datetime(df["DISCOVERYDATETIME"], format="%m/%d/%y %H:%M", errors="coerce")
    df["date"] = df["date"].apply(lambda x: x.replace(year=x.year - 100) if pd.notnull(x) and x.year > 2000 else x)
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date"])
    df = df[df["FIRESEASON"].astype(str).str.strip().isin([str(y) for y in YEARS])]
    df = df[df["date"].dt.month.isin([int(m) for m in MONTHS])]
    print(f"[INFO] Wildfire records in study window: {len(df)}")
    
    df["cause_lower"] = df["SPECIFICCAUSE"].str.lower().str.strip().fillna("undetermined")
    df["weight"] = df["cause_lower"].map(
        lambda c: next((v for k, v in CAUSE_WEIGHTS.items() if k in str(c)), DEFAULT_WEIGHT)
    )
    
    # Daily risk score = sum of weights for all fires that day
    # A day with 3 lightning fires scores higher than a day with 1 human fire
    risk = df.groupby("date")["weight"].sum().reset_index()
    risk = risk.rename(columns={"weight": "risk_score"})
    
    # Normalize risk score to 0-1 range for better model training
    max_score = risk["risk_score"].max()
    if max_score > 0:
        risk["risk_score"] = risk["risk_score"] / max_score
        
    print(f"[INFO] Wildfire DataFrame shape: {risk.shape}")
    return risk


def build_dataset() -> pd.DataFrame:
    if not os.path.exists(WEATHER_DATA_DIR):
        raise FileNotFoundError(f"Weather data directory not found: {WEATHER_DATA_DIR}")
    if not os.path.exists(WILDFIRE_DATA_DIR):
        raise FileNotFoundError(f"Wildfire CSV not found: {WILDFIRE_DATA_DIR}")
    
    weather = load_weather(WEATHER_DATA_DIR)
    risk    = load_risk_scores(WILDFIRE_DATA_DIR)
 
    weather["date"] = pd.to_datetime(weather["date"])
    risk["date"] = pd.to_datetime(risk["date"])
    df = weather.merge(risk, on="date", how="left")

    df["risk_score"] = df["risk_score"].fillna(0.0)
    df = df.dropna()
 
    # Save to OUTPUT_CSV for later use in model training
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Saved ML-ready dataset to: {OUTPUT_CSV}")
    print(f"[INFO] Final shape: {df.shape}")
    print(f"\n{df.head()}")
    return df
 
 
if __name__ == "__main__":
    df = build_dataset()

