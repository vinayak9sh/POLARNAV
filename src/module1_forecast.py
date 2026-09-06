from pathlib import Path

import numpy as np
import pandas as pd


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
