
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

LAT_MODEL_PATH = MODEL_DIR / "module2_final_rf_latitude.joblib"
LON_MODEL_PATH = MODEL_DIR / "module2_final_rf_longitude.joblib"
TRAJECTORY_PATH = DATA_DIR / "module2_ascat_trajectories.parquet"


FEATURES = [
    "latitude",
    "longitude",
    "speed_km_day",
    "heading_sin",
    "heading_cos",
    "speed_lag1",
    "speed_lag2",
    "speed_lag3",
    "heading_lag1_sin",
    "heading_lag1_cos",
    "heading_lag2_sin",
    "heading_lag2_cos",
    "heading_lag3_sin",
    "heading_lag3_cos",
]


# ------------------------------------------------------------
# Load application assets
# ------------------------------------------------------------

if not LAT_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Latitude model not found: {LAT_MODEL_PATH}"
    )

if not LON_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Longitude model not found: {LON_MODEL_PATH}"
    )

if not TRAJECTORY_PATH.exists():
    raise FileNotFoundError(
        f"Trajectory dataset not found: {TRAJECTORY_PATH}"
    )


LAT_MODEL = joblib.load(LAT_MODEL_PATH)
LON_MODEL = joblib.load(LON_MODEL_PATH)

TRAJECTORY_DF = pd.read_parquet(
    TRAJECTORY_PATH
)

TRAJECTORY_DF["date"] = pd.to_datetime(
    TRAJECTORY_DF["date"]
)

TRAJECTORY_DF = (
    TRAJECTORY_DF
    .sort_values(["iceberg_id", "date"])
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Geographic helpers
# ------------------------------------------------------------

def _normalize_longitude(longitude):
    return ((longitude + 180.0) % 360.0) - 180.0


def _shortest_longitude_difference(lon2, lon1):
    return ((lon2 - lon1 + 180.0) % 360.0) - 180.0


def _haversine_km(lat1, lon1, lat2, lon2):

    radius_km = 6371.0088

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = lat2_rad - lat1_rad

    dlon = math.radians(
        _shortest_longitude_difference(
            lon2,
            lon1
        )
    )

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2.0) ** 2
    )

    return (
        radius_km
        * 2.0
        * math.asin(
            math.sqrt(
                min(1.0, max(0.0, a))
            )
        )
    )


def _bearing_deg(lat1, lon1, lat2, lon2):

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlon_rad = math.radians(
        _shortest_longitude_difference(
            lon2,
            lon1
        )
    )

    x = (
        math.sin(dlon_rad)
        * math.cos(lat2_rad)
    )

    y = (
        math.cos(lat1_rad)
        * math.sin(lat2_rad)
        - math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(dlon_rad)
    )

    return (
        math.degrees(
            math.atan2(x, y)
        ) % 360.0
    )


def _circular_components(heading_deg):

    angle_rad = math.radians(
        heading_deg
    )

    return (
        math.sin(angle_rad),
        math.cos(angle_rad)
    )


# ------------------------------------------------------------
# Build movement history
# ------------------------------------------------------------

def _build_motion_history(
    iceberg_id,
    forecast_date
):

    forecast_date = pd.Timestamp(
        forecast_date
    )

    iceberg = TRAJECTORY_DF[
        TRAJECTORY_DF["iceberg_id"] == iceberg_id
    ].copy()

    if iceberg.empty:
        raise ValueError(
            f"Iceberg '{iceberg_id}' was not found."
        )

    iceberg = iceberg[
        iceberg["date"] <= forecast_date
    ].copy()

    if iceberg.empty:
        raise ValueError(
            f"No observations exist for iceberg "
            f"'{iceberg_id}' on or before "
            f"{forecast_date:%Y-%m-%d}."
        )

    iceberg["prev_latitude"] = (
        iceberg["latitude"].shift(1)
    )

    iceberg["prev_longitude"] = (
        iceberg["longitude"].shift(1)
    )

    iceberg["prev_date"] = (
        iceberg["date"].shift(1)
    )

    iceberg["days_since_prev"] = (
        iceberg["date"] - iceberg["prev_date"]
    ).dt.total_seconds() / 86400.0

    # Movement distance
    lat1 = np.radians(
        iceberg["prev_latitude"]
    )

    lat2 = np.radians(
        iceberg["latitude"]
    )

    dlat = lat2 - lat1

    dlon = np.radians(
        (
            iceberg["longitude"]
            - iceberg["prev_longitude"]
            + 180.0
        ) % 360.0 - 180.0
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    iceberg["distance_km"] = (
        6371.0088
        * 2.0
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )

    iceberg["speed_km_day"] = (
        iceberg["distance_km"]
        / iceberg["days_since_prev"]
    )

    # Heading
    lat1_rad = np.radians(
        iceberg["prev_latitude"]
    )

    lat2_rad = np.radians(
        iceberg["latitude"]
    )

    heading_dlon = np.radians(
        (
            iceberg["longitude"]
            - iceberg["prev_longitude"]
            + 180.0
        ) % 360.0 - 180.0
    )

    x = (
        np.sin(heading_dlon)
        * np.cos(lat2_rad)
    )

    y = (
        np.cos(lat1_rad)
        * np.sin(lat2_rad)
        - np.sin(lat1_rad)
        * np.cos(lat2_rad)
        * np.cos(heading_dlon)
    )

    iceberg["heading_deg"] = (
        np.degrees(
            np.arctan2(x, y)
        ) % 360.0
    )

    # Same daily-transition requirement as training
    iceberg = iceberg[
        iceberg["days_since_prev"] == 1
    ].copy()

    # Same movement QC threshold as training
    iceberg = iceberg[
        iceberg["speed_km_day"] <= 50
    ].copy()

    # Lag features
    for lag in [1, 2, 3]:

        iceberg[f"speed_lag{lag}"] = (
            iceberg["speed_km_day"]
            .shift(lag)
        )

        iceberg[f"heading_lag{lag}"] = (
            iceberg["heading_deg"]
            .shift(lag)
        )

    return (
        iceberg
        .sort_values("date")
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------

def _build_features(history):

    history = (
        history
        .sort_values("date")
        .tail(4)
        .copy()
    )

    if len(history) != 4:
        raise ValueError(
            "Exactly four valid movement observations "
            "are required for prediction."
        )

    date_diffs = (
        history["date"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 86400.0
    )

    if not np.allclose(
        date_diffs.to_numpy(),
        1.0
    ):
        raise ValueError(
            "The four latest movement observations "
            "must be consecutive daily observations."
        )

    latest = history.iloc[-1]
    lag1 = history.iloc[-2]
    lag2 = history.iloc[-3]
    lag3 = history.iloc[-4]

    h0_sin, h0_cos = _circular_components(
        float(latest["heading_deg"])
    )

    h1_sin, h1_cos = _circular_components(
        float(lag1["heading_deg"])
    )

    h2_sin, h2_cos = _circular_components(
        float(lag2["heading_deg"])
    )

    h3_sin, h3_cos = _circular_components(
        float(lag3["heading_deg"])
    )

    values = {
        "latitude": float(latest["latitude"]),
        "longitude": float(latest["longitude"]),
        "speed_km_day": float(
            latest["speed_km_day"]
        ),

        "heading_sin": h0_sin,
        "heading_cos": h0_cos,

        "speed_lag1": float(
            lag1["speed_km_day"]
        ),
        "speed_lag2": float(
            lag2["speed_km_day"]
        ),
        "speed_lag3": float(
            lag3["speed_km_day"]
        ),

        "heading_lag1_sin": h1_sin,
        "heading_lag1_cos": h1_cos,

        "heading_lag2_sin": h2_sin,
        "heading_lag2_cos": h2_cos,

        "heading_lag3_sin": h3_sin,
        "heading_lag3_cos": h3_cos,
    }

    return pd.DataFrame(
        [values],
        columns=FEATURES
    )


# ------------------------------------------------------------
# Public API functions
# ------------------------------------------------------------

def list_icebergs():

    return sorted(
        TRAJECTORY_DF[
            "iceberg_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_available_dates(iceberg_id):

    iceberg = TRAJECTORY_DF[
        TRAJECTORY_DF["iceberg_id"] == iceberg_id
    ]

    if iceberg.empty:
        raise ValueError(
            f"Iceberg '{iceberg_id}' was not found."
        )

    return {
        "iceberg_id": iceberg_id,
        "first_date": iceberg["date"].min().strftime(
            "%Y-%m-%d"
        ),
        "last_date": iceberg["date"].max().strftime(
            "%Y-%m-%d"
        ),
        "observation_count": int(
            len(iceberg)
        )
    }


def get_iceberg_history(
    iceberg_id,
    forecast_date
):

    motion = _build_motion_history(
        iceberg_id,
        forecast_date
    )

    if len(motion) < 4:
        raise ValueError(
            f"Iceberg '{iceberg_id}' does not have "
            f"four valid movement observations before "
            f"{pd.Timestamp(forecast_date):%Y-%m-%d}."
        )

    history = motion.tail(4).copy()

    latest_date = pd.Timestamp(
        history.iloc[-1]["date"]
    )

    requested_date = pd.Timestamp(
        forecast_date
    )

    if latest_date != requested_date:
        raise ValueError(
            f"Iceberg '{iceberg_id}' has no valid "
            f"trajectory observation on "
            f"{requested_date:%Y-%m-%d}. "
            f"Latest usable observation is "
            f"{latest_date:%Y-%m-%d}."
        )

    return history


def predict_iceberg(
    iceberg_id,
    forecast_date
):

    forecast_date = pd.Timestamp(
        forecast_date
    )

    history = get_iceberg_history(
        iceberg_id,
        forecast_date
    )

    latest = history.iloc[-1]

    features = _build_features(
        history
    )

    delta_lat = float(
        LAT_MODEL.predict(features)[0]
    )

    delta_lon = float(
        LON_MODEL.predict(features)[0]
    )

    current_lat = float(
        latest["latitude"]
    )

    current_lon = float(
        latest["longitude"]
    )

    predicted_lat = (
        current_lat + delta_lat
    )

    predicted_lon = _normalize_longitude(
        current_lon + delta_lon
    )

    distance_km = _haversine_km(
        current_lat,
        current_lon,
        predicted_lat,
        predicted_lon
    )

    bearing_deg = _bearing_deg(
        current_lat,
        current_lon,
        predicted_lat,
        predicted_lon
    )

    return {
        "iceberg_id": str(iceberg_id),
        "input_date": forecast_date.strftime(
            "%Y-%m-%d"
        ),
        "prediction_date": (
            forecast_date
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "current_latitude": current_lat,
        "current_longitude": current_lon,
        "predicted_latitude": predicted_lat,
        "predicted_longitude": predicted_lon,
        "predicted_distance_km": distance_km,
        "predicted_bearing_deg": bearing_deg,
        "predicted_delta_lat": delta_lat,
        "predicted_delta_lon": delta_lon,
    }



def get_coverage_aware_predictions(
    forecast_date,
    max_persistence_days=3,
    persistence_hazard_weight=0.25
):
    """
    Generate date-specific iceberg position estimates.

    Prediction methods:
        ml_random_forest
            Validated Random Forest next-day prediction.

        persistence
            Last observed position used when the latest
            observation is within max_persistence_days.

        excluded
            Track is too old or has no observation.

    Persistence is explicitly NOT an ML prediction.
    """

    forecast_date = pd.Timestamp(
        forecast_date
    )

    max_persistence_days = int(
        max_persistence_days
    )

    persistence_hazard_weight = float(
        persistence_hazard_weight
    )

    if max_persistence_days < 0:
        raise ValueError(
            "max_persistence_days must be >= 0."
        )

    if not (
        0.0 <= persistence_hazard_weight <= 1.0
    ):
        raise ValueError(
            "persistence_hazard_weight must be "
            "between 0 and 1."
        )

    results = []
    excluded = []

    iceberg_ids = list_icebergs()

    for iceberg_id in iceberg_ids:

        # ----------------------------------------------------
        # 1. Try validated Random Forest prediction
        # ----------------------------------------------------
        try:

            result = predict_iceberg(
                iceberg_id,
                forecast_date.strftime("%Y-%m-%d")
            )

            result["prediction_method"] = (
                "ml_random_forest"
            )

            result["confidence"] = "high"

            result["observation_age_days"] = 0

            result["hazard_weight"] = 1.0

            results.append(result)

            continue

        except ValueError:
            # ML unavailable for this track/date.
            pass

        # ----------------------------------------------------
        # 2. Find latest observation
        # ----------------------------------------------------
        track = TRAJECTORY_DF[
            TRAJECTORY_DF["iceberg_id"]
            == iceberg_id
        ]

        history = track[
            track["date"] <= forecast_date
        ]

        if history.empty:

            excluded.append({
                "iceberg_id": iceberg_id,
                "reason": (
                    "no_observation_before_forecast_date"
                )
            })

            continue

        latest = (
            history
            .sort_values("date")
            .iloc[-1]
        )

        observation_age_days = int(
            (
                forecast_date
                - pd.Timestamp(
                    latest["date"]
                )
            ).days
        )

        # ----------------------------------------------------
        # 3. Persistence fallback
        # ----------------------------------------------------
        if observation_age_days <= max_persistence_days:

            results.append({
                "iceberg_id": str(iceberg_id),

                "input_date": forecast_date.strftime(
                    "%Y-%m-%d"
                ),

                "prediction_date": (
                    forecast_date
                    + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),

                "current_latitude": float(
                    latest["latitude"]
                ),

                "current_longitude": float(
                    latest["longitude"]
                ),

                "predicted_latitude": float(
                    latest["latitude"]
                ),

                "predicted_longitude": float(
                    latest["longitude"]
                ),

                "predicted_distance_km": 0.0,

                "predicted_bearing_deg": None,

                "predicted_delta_lat": 0.0,

                "predicted_delta_lon": 0.0,

                "prediction_method": (
                    "persistence"
                ),

                "confidence": "low",

                "observation_age_days": (
                    observation_age_days
                ),

                "hazard_weight": (
                    persistence_hazard_weight
                )
            })

        else:

            excluded.append({
                "iceberg_id": str(iceberg_id),

                "reason": (
                    f"latest observation is "
                    f"{observation_age_days} days old"
                ),

                "observation_age_days": (
                    observation_age_days
                )
            })

    total_tracks = len(iceberg_ids)

    ml_count = sum(
        p["prediction_method"]
        == "ml_random_forest"
        for p in results
    )

    persistence_count = sum(
        p["prediction_method"]
        == "persistence"
        for p in results
    )

    represented_count = len(results)

    excluded_count = len(excluded)

    coverage_percent = (
        100.0
        * represented_count
        / total_tracks
        if total_tracks > 0
        else 0.0
    )

    return {
        "forecast_date": forecast_date.strftime(
            "%Y-%m-%d"
        ),

        "prediction_date": (
            forecast_date
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),

        "predictions": results,

        "excluded": excluded,

        "coverage": {
            "total_tracks": total_tracks,
            "ml_predictions": ml_count,
            "persistence_estimates": persistence_count,
            "represented_tracks": represented_count,
            "excluded_tracks": excluded_count,
            "coverage_percent": round(
                coverage_percent,
                2
            )
        }
    }

