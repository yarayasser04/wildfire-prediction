import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import socket

WEATHER_DATA_DIR = "/Users/yarayasser/capstone_wildfire_prediction/wildfire_prediction/data/weather_data"
WILDFIRE_DATA_DIR = "/Users/yarayasser/capstone_wildfire_prediction/wildfire_prediction/data/wildfire_data/AK_fire_location_points_NAD83.csv"
OUTPUT_CSV = "/Users/yarayasser/capstone_wildfire_prediction/wildfire_prediction/data/ml_ready.csv"

YEARS = range(1950, 1953)
MONTHS = ["05", "06", "07", "08"]

# Risk weight per cause — higher = riskier
# Lightning = highest risk (natural, unpredictable, hardest to control)
# Human/Structure = moderate (preventable but still dangerous)
# Hand pile/slash = lower (controlled burns, usually managed)
# Undetermined = low default since cause is unknown
CAUSE_WEIGHTS = {
    "lightning":                              1.0,  # natural, unpredictable, hardest to control
    "electrical transmission/distribution":   0.8,  # can ignite large areas quickly
    "military ordnance":                      0.7,  # explosive ignition, hard to predict
    "human":                                  0.7,  # general human cause
    "campfire":                               0.6,  # negligent but localized
    "structure":                              0.6,  # can spread but usually contained
    "passenger vehicle/motorized rv":         0.5,  # roadside ignition, moderate spread
    "hand pile/slash":                        0.3,  # typically managed/controlled burns
    "not investigated":                       0.2,  # unknown, low default
    "undetermined":                           0.2,  # unknown, low default
}
DEFAULT_WEIGHT = 0.2  # used for any other cause not listed above

def weather_get_data_dir() -> str:
    if os.path.exists(WEATHER_DATA_DIR):
        return WEATHER_DATA_DIR
    raise FileNotFoundError(
        f"Data directory not found: {WEATHER_DATA_DIR}\n"
        f"Please update DATA_DIR at the top of this script."
    )
def wildfire_get_data_dir() -> str:
    if os.path.exists(WILDFIRE_DATA_DIR):
        return WILDFIRE_DATA_DIR
    raise FileNotFoundError(
        f"Data directory not found: {WILDFIRE_DATA_DIR}\n"
        f"Please update DATA_DIR at the top of this script."
    )
    
# Load weather data files based on specified years and months, ensuring they exist
def get_files(data_dir: str) -> list[str]:
    files = []
    for year in YEARS:
        for month in MONTHS:
            pattern = os.path.join(data_dir, f"fairbanks_{year}_{month}*.nc")
            matched = sorted(glob.glob(pattern))
            if not matched:
                print(f"[WARNING] No files found for {year}-{month}")
            files.extend(matched)
    return files

def load_weather(data_dir: str) -> pd.DataFrame:
    files = get_files(data_dir)
    print(f"[INFO] Found {len(files)} weather files")
    
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4", chunks="auto")
    
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
 