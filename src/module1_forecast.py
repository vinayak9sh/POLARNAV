from pathlib import Path

import numpy as np
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parent.parent

# Final multi-year Module 1 model
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "module1_final_random_forest.joblib"
)

# Compact runtime used by the API
RUNTIME_PATH = (
    BASE_DIR
    / "models"
    / "module1_outputs"
    / "module1_final_forecast_runtime.npz"
)

LONG_RANGE_CONFIG_PATH = (
    BASE_DIR
    / "long_range_forecast_config.json"
)

FEATURES = [
    "lat",
    "lon",
    "sea_ice_concentration",
    "sic_lag1",
    "sic_lag2",
]


def load_forecast_runtime():
    """
    Load the compact SIC history and spatial grid.
    """

    if not RUNTIME_PATH.exists():
        raise FileNotFoundError(
            f"SIC runtime not found: {RUNTIME_PATH}"
        )

    data = np.load(
        RUNTIME_PATH,
        allow_pickle=False
    )

    return {
        "sic_history": data["sic_history"],
        "dates": pd.to_datetime(data["dates"]),
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "yc": data["yc"],
        "xc": data["xc"],
    }

def load_long_range_history():
    """
    Load the regridded monthly SIC history used by the
    validated long-range hindcast system.

    Returns
    -------
    dict
        Monthly SIC history and corresponding dates.
    """

    history_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "nsidc_monthly_2018_2025_polarnav.nc"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            "Long-range SIC history not found: "
            f"{history_path}"
        )

    import xarray as xr

    ds = xr.open_dataset(history_path)

    sic = ds["sic"].to_numpy().astype(
    np.float32
)

    dates = pd.to_datetime(
        ds["time"].to_numpy()
    )

    ds.close()

    return {
        "sic": sic,
        "dates": dates,
    }

def calculate_seasonal_climatology(target_date):
    """
    Calculate the historical seasonal SIC climatology
    for the month of the requested target date.

    The climatology is calculated independently for
    each POLARNAV grid cell.

    Parameters
    ----------
    target_date : str or pandas.Timestamp

    Returns
    -------
    dict
        Monthly climatological SIC map and metadata.
    """

    history = load_long_range_history()

    sic_history = history["sic"]
    dates = history["dates"]

    target_date = pd.Timestamp(target_date)

    target_month = target_date.month

    # --------------------------------------------------
    # Select historical observations belonging to the
    # same calendar month.
    # --------------------------------------------------

    month_mask = (
        dates.month == target_month
    )

    monthly_history = sic_history[
        month_mask
    ]

    if monthly_history.shape[0] == 0:
        raise ValueError(
            f"No historical SIC data available "
            f"for month {target_month}."
        )

    # --------------------------------------------------
    # Average independently at every spatial cell.
    # NaNs are ignored.
    # --------------------------------------------------

    valid_count = np.sum(
        np.isfinite(monthly_history),
        axis=0
    )

    climatology = np.full(
        monthly_history.shape[1:],
        np.nan,
        dtype=np.float32
    )

    valid_cells = valid_count > 0

    climatology[valid_cells] = (
        np.nansum(
            monthly_history[:, valid_cells],
            axis=0
        )
        / valid_count[valid_cells]
    ).astype(np.float32)

    return {
        "target_date": target_date,
        "target_month": target_month,
        "climatology": climatology,
        "historical_cases":
            int(monthly_history.shape[0]),
    }

def calculate_anomaly_adjustment(
    latest_sic,
    latest_date,
    target_date,
    climatology
):
    """
    Calculate the anomaly-aware SIC forecast adjustment.

    The adjustment is:

        forecast = target climatology
                    + alpha * current anomaly

    where:

        current anomaly =
            latest observed SIC
            - climatology for the latest month

    Parameters
    ----------
    latest_sic : np.ndarray
        Latest observed SIC map on the POLARNAV grid,
        expressed as a fraction (0-1).

    latest_date : str or pandas.Timestamp
        Date corresponding to latest_sic.

    target_date : str or pandas.Timestamp
        Requested forecast date.

    climatology : np.ndarray
        Historical seasonal climatology for the target month.

    Returns
    -------
    dict
        Anomaly map, alpha, and adjustment map.
    """

    latest_date = pd.Timestamp(latest_date)
    target_date = pd.Timestamp(target_date)

    horizon_months = calculate_forecast_horizon(
        target_date,
        latest_date
    )

    # --------------------------------------------------
    # Load validated alpha values.
    # --------------------------------------------------

    alpha_by_lead = {
        1: 0.65,
        2: 0.15,
        3: 0.05,
        4: 0.0,
        5: 0.0,
        6: 0.0,
    }

    if horizon_months not in alpha_by_lead:
        raise ValueError(
            "Unsupported anomaly-adjustment horizon: "
            f"{horizon_months} months."
        )

    alpha = alpha_by_lead[horizon_months]

    # --------------------------------------------------
    # Find the climatology corresponding to the latest
    # observed month.
    # --------------------------------------------------

    latest_climatology_result = (
        calculate_seasonal_climatology(
            latest_date
        )
    )

    latest_climatology = (
        latest_climatology_result["climatology"]
        * 100.0
    ).astype(np.float32)

    target_climatology = (
        climatology
        * 100.0
    ).astype(np.float32)

    # --------------------------------------------------
    # Calculate current spatial anomaly.
    # --------------------------------------------------

    valid = (
        np.isfinite(latest_sic)
        & np.isfinite(latest_climatology)
        & np.isfinite(target_climatology)
    )

    anomaly = np.full(
        latest_sic.shape,
        np.nan,
        dtype=np.float32
    )

    anomaly[valid] = (
        latest_sic[valid]
        - latest_climatology[valid]
    )

    adjustment = np.zeros(
        latest_sic.shape,
        dtype=np.float32
    )

    adjustment[valid] = (
        alpha * anomaly[valid]
    )

    return {
        "horizon_months": horizon_months,
        "alpha": alpha,
        "latest_climatology": latest_climatology,
        "target_climatology": target_climatology,
        "anomaly": anomaly,
        "adjustment": adjustment,
    }

def forecast_long_range_sic(forecast_date):
    """
    Generate a six-month long-range SIC forecast.

    Forecast strategy:
        +1 to +3 months -> anomaly-aware
        +4 to +6 months -> seasonal climatology

    The latest available observed SIC map is used as the
    starting state.

    Returns
    -------
    dict
        Forecast map, method, horizon, uncertainty,
        confidence, and spatial coordinates.
    """

    forecast_date = pd.Timestamp(
        forecast_date
    )

    # --------------------------------------------------
    # Load latest observed POLARNAV SIC
    # --------------------------------------------------

    runtime = load_forecast_runtime()

    latest_date = pd.Timestamp(
        runtime["dates"][-1]
    )

    latest_sic = runtime[
        "sic_history"
    ][-1].astype(np.float32)

    latitude = runtime["latitude"]
    longitude = runtime["longitude"]

    # --------------------------------------------------
    # Validate requested date
    # --------------------------------------------------

    (
        forecast_date,
        horizon_months,
        method,
    ) = validate_long_range_date(
        forecast_date,
        latest_date
    )

    # --------------------------------------------------
    # Target-month seasonal climatology
    # --------------------------------------------------

    climatology_result = (
        calculate_seasonal_climatology(
            forecast_date
        )
    )

    target_climatology = (
        climatology_result["climatology"]
    )

    # --------------------------------------------------
    # Forecast construction
    # --------------------------------------------------

    if method == "climatology":

        forecast_map = (
            target_climatology * 100.0
        ).astype(np.float32)

        alpha = 0.0
        anomaly = None

    else:

        anomaly_result = (
            calculate_anomaly_adjustment(
                latest_sic=latest_sic,
                latest_date=latest_date,
                target_date=forecast_date,
                climatology=target_climatology,
            )
        )

        alpha = anomaly_result["alpha"]
        anomaly = anomaly_result["anomaly"]

        forecast_map = (
            target_climatology * 100.0
            + anomaly_result["adjustment"]
        ).astype(np.float32)

    # --------------------------------------------------
    # Keep SIC physically meaningful.
    # --------------------------------------------------

    forecast_map = np.clip(
        forecast_map,
        0.0,
        100.0
    ).astype(np.float32)

    # --------------------------------------------------
    # Load empirical uncertainty.
    # --------------------------------------------------

    config = load_long_range_config()

    uncertainty = config[
        "uncertainty_percent_sic"
    ][str(horizon_months)]

    confidence = config[
        "confidence_level_by_lead_month"
    ][str(horizon_months)]

    # --------------------------------------------------
    # Valid output cells
    # --------------------------------------------------

    valid = np.isfinite(
        forecast_map
    )

    return {
    "forecast_date": forecast_date,
    "latest_observation_date": latest_date,
    "horizon_months": horizon_months,
    "method": method,
    "alpha": alpha,

    # Long-range-specific name
    "forecast_sic": forecast_map,

    # Compatibility with the existing navigation pipeline
    "predicted_sic": forecast_map,

    # For the existing API/route response structure,
    # the target date itself is the forecast date.
    "prediction_date": forecast_date,

    "target_climatology": (
        target_climatology * 100.0
    ).astype(np.float32),

    "anomaly": anomaly,
    "uncertainty": uncertainty,
    "confidence": confidence,
    "valid_cells": int(valid.sum()),
    "latitude": latitude,
    "longitude": longitude,
    "yc": runtime["yc"],
    "xc": runtime["xc"],
}

def load_forecast_model():
    """
    Load the final multi-year Module 1 model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    import joblib

    return joblib.load(MODEL_PATH)

def load_long_range_config():
    """
    Load the validated six-month forecast configuration.
    """

    if not LONG_RANGE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Long-range forecast config not found: "
            f"{LONG_RANGE_CONFIG_PATH}"
        )

    with open(
        LONG_RANGE_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def calculate_forecast_horizon(forecast_date, latest_date):
    """
    Calculate whole calendar-month forecast horizon.

    A date exactly one calendar month after the latest
    observation has horizon 1. A date exactly three
    calendar months after it has horizon 3.

    Dates between calendar-month boundaries are assigned
    to the next month so that the selected forecast always
    has a conservative lead-time classification.
    """

    forecast_date = pd.Timestamp(forecast_date)
    latest_date = pd.Timestamp(latest_date)

    if forecast_date <= latest_date:
        return 0

    # Calendar month difference
    months = (
        (forecast_date.year - latest_date.year) * 12
        + (forecast_date.month - latest_date.month)
    )

    # Same month but later date → one-month horizon
    if months == 0:
        return 1

    # If the target day is beyond the corresponding day
    # in the latest month, count the partial month as the
    # next lead month.
    if forecast_date.day > latest_date.day:
        months += 1

    return max(1, months)

def get_long_range_method(horizon_months):
    """
    Select the validated forecasting method for a
    given lead time.
    """

    config = load_long_range_config()

    if horizon_months < 1:
        return "observed"

    if horizon_months > config["max_forecast_months"]:
        raise ValueError(
            "Forecast horizon exceeds the supported "
            f"{config['max_forecast_months']}-month window."
        )

    return config["method_by_lead_month"][
        str(horizon_months)
    ]

def validate_long_range_date(forecast_date, latest_date):
    """
    Validate that a requested forecast date is inside the
    supported six-month forecast window.

    Returns
    -------
    tuple
        (forecast_date, horizon_months, method)
    """

    forecast_date = pd.Timestamp(forecast_date)
    latest_date = pd.Timestamp(latest_date)

    if forecast_date <= latest_date:
        raise ValueError(
            "Long-range forecast date must be after "
            f"the latest observed date {latest_date.date()}."
        )

    horizon_months = calculate_forecast_horizon(
        forecast_date,
        latest_date
    )

    config = load_long_range_config()

    if horizon_months > config["max_forecast_months"]:
        max_date = (
            latest_date
            + pd.DateOffset(
                months=config["max_forecast_months"]
            )
        )

        raise ValueError(
            "Requested forecast date exceeds the "
            f"{config['max_forecast_months']}-month "
            "forecast window. "
            f"Latest observation: {latest_date.date()}. "
            f"Maximum supported date: {max_date.date()}."
        )

    method = get_long_range_method(horizon_months)

    return (
        forecast_date,
        horizon_months,
        method
    )

def get_long_range_forecast_window():
    """
    Return the currently supported long-range forecast window.

    The window is anchored to the latest observation
    available in the POLARNAV runtime asset.
    """

    runtime = load_forecast_runtime()

    latest_date = pd.Timestamp(
        runtime["dates"][-1]
    )

    config = load_long_range_config()

    max_months = int(
        config["max_forecast_months"]
    )

    maximum_date = (
        latest_date
        + pd.DateOffset(
            months=max_months
        )
    )

    return {
        "latest_observation_date": latest_date,
        "maximum_forecast_date": maximum_date,
        "max_forecast_months": max_months,
    }

def forecast_sic(forecast_date):
    """
    Predict next-day SIC using:

        current SIC
        SIC lag-1
        SIC lag-2
        latitude
        longitude

    The requested date and its two previous dates must
    be consecutive calendar days.

    Returns
    -------
    dict
        Contains dataframe output plus spatial arrays.
    """

    runtime = load_forecast_runtime()
    model = load_forecast_model()

    forecast_date = pd.Timestamp(
        forecast_date
    )

    dates = runtime["dates"]

    # --------------------------------------------------
    # Locate requested date
    # --------------------------------------------------

    matches = np.where(
        dates == forecast_date
    )[0]

    if len(matches) == 0:
        raise ValueError(
            "Forecast date is not available. "
            f"Available dates: {dates[0].date()} "
            f"to {dates[-1].date()}."
        )

    idx = int(matches[0])

    # --------------------------------------------------
    # Need two previous observations
    # --------------------------------------------------

    if idx < 2:
        raise ValueError(
            "At least two previous SIC days are "
            "required for forecasting."
        )

    # --------------------------------------------------
    # IMPORTANT: enforce temporal continuity
    #
    # This prevents a gap such as:
    # 2025-03-17 → 2025-03-19
    # from being treated as lag-1.
    # --------------------------------------------------

    expected_lag1 = (
        forecast_date
        - pd.Timedelta(days=1)
    )

    expected_lag2 = (
        forecast_date
        - pd.Timedelta(days=2)
    )

    actual_lag1 = dates[idx - 1]
    actual_lag2 = dates[idx - 2]

    if (
        actual_lag1 != expected_lag1
        or actual_lag2 != expected_lag2
    ):
        raise ValueError(
            "Forecast cannot be generated because "
            "the required SIC history is not "
            "temporally continuous. "
            f"Required: {expected_lag2.date()}, "
            f"{expected_lag1.date()}, "
            f"{forecast_date.date()}. "
            f"Available: {actual_lag2.date()}, "
            f"{actual_lag1.date()}, "
            f"{forecast_date.date()}."
        )

    # --------------------------------------------------
    # Current and lagged SIC
    # --------------------------------------------------

    current_sic = runtime["sic_history"][idx]
    sic_lag1 = runtime["sic_history"][idx - 1]
    sic_lag2 = runtime["sic_history"][idx - 2]

    lat = runtime["latitude"]
    lon = runtime["longitude"]

    # --------------------------------------------------
    # Valid spatial cells
    # --------------------------------------------------

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(current_sic)
        & np.isfinite(sic_lag1)
        & np.isfinite(sic_lag2)
    )

    rows, cols = np.where(valid)

    # --------------------------------------------------
    # Build model input
    # --------------------------------------------------

    input_df = pd.DataFrame({
        "lat": lat[valid].astype("float32"),
        "lon": lon[valid].astype("float32"),
        "sea_ice_concentration":
            current_sic[valid].astype("float32"),
        "sic_lag1":
            sic_lag1[valid].astype("float32"),
        "sic_lag2":
            sic_lag2[valid].astype("float32"),
    })

    # The model was trained with NumPy arrays,
    # so pass a NumPy matrix here to avoid
    # sklearn feature-name warnings.
    X = input_df[FEATURES].to_numpy(
        dtype="float32"
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = np.clip(
        model.predict(X),
        0,
        100
    ).astype("float32")

    input_df["predicted_sic"] = predictions

    # --------------------------------------------------
    # Reconstruct spatial prediction grid
    # --------------------------------------------------

    predicted_map = np.full(
        current_sic.shape,
        np.nan,
        dtype="float32"
    )

    predicted_map[
        rows,
        cols
    ] = predictions

    return {
        "forecast_date": forecast_date,
        "prediction_date":
            forecast_date + pd.Timedelta(days=1),
        "dataframe": input_df,
        "predicted_sic": predicted_map,
        "latitude": lat,
        "longitude": lon,
        "yc": runtime["yc"],
        "xc": runtime["xc"],
    }
