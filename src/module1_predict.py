
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "module1_30day_mvp_random_forest.joblib"
)

FEATURES = [
    "lat",
    "lon",
    "sea_ice_concentration",
    "sic_lag1",
    "sic_lag2",
]


def load_model():
    """Load the trained Module 1 MVP Random Forest."""
    return joblib.load(MODEL_PATH)


def predict_next_day_sic(
    input_df: pd.DataFrame,
    model=None
) -> pd.DataFrame:
    """
    Predict next-day sea-ice concentration.

    Required columns:
        lat
        lon
        sea_ice_concentration
        sic_lag1
        sic_lag2
    """

    missing = [
        column for column in FEATURES
        if column not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if model is None:
        model = load_model()

    result = input_df.copy()

    if result[FEATURES].isna().any().any():
        raise ValueError(
            "Input contains missing values in required features."
        )

    X = result[FEATURES].astype("float32")

    predictions = model.predict(X)

    result["predicted_sic"] = np.clip(
        predictions,
        0,
        100
    )

    return result


def create_spatial_output(result_df, yc_values, xc_values):
    """
    Convert Module 1 dataframe output into native grid arrays.

    Returns a dictionary containing:
        predicted_sic
        risk_code
        latitude
        longitude
        yc
        xc
    """

    import numpy as np

    yc_values = np.asarray(yc_values)
    xc_values = np.asarray(xc_values)

    predicted_sic = np.full(
        (len(yc_values), len(xc_values)),
        np.nan,
        dtype=np.float32
    )

    risk_code = np.full(
        (len(yc_values), len(xc_values)),
        -1,
        dtype=np.int8
    )

    latitude = np.full(
        (len(yc_values), len(xc_values)),
        np.nan,
        dtype=np.float32
    )

    longitude = np.full(
        (len(yc_values), len(xc_values)),
        np.nan,
        dtype=np.float32
    )

    yc_index = {
        value: i
        for i, value in enumerate(yc_values)
    }

    xc_index = {
        value: i
        for i, value in enumerate(xc_values)
    }

    for row in result_df.itertuples(index=False):

        if row.yc not in yc_index or row.xc not in xc_index:
            continue

        i = yc_index[row.yc]
        j = xc_index[row.xc]

        predicted_sic[i, j] = row.predicted_sic
        risk_code[i, j] = row.risk_code
        latitude[i, j] = row.lat
        longitude[i, j] = row.lon

    return {
        "predicted_sic": predicted_sic,
        "risk_code": risk_code,
        "latitude": latitude,
        "longitude": longitude,
        "yc": yc_values,
        "xc": xc_values,
    }
