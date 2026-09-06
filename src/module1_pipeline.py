
import numpy as np
import pandas as pd

from module1_predict import (
    load_model,
    predict_next_day_sic,
    create_spatial_output
)

from module1_risk import (
    classify_sic_risk,
    risk_labels
)


def run_module1(input_df, model=None):
    """
    Run the complete Module 1 pipeline.

    Input columns:
        lat
        lon
        sea_ice_concentration
        sic_lag1
        sic_lag2

    Returns:
        DataFrame containing:
        predicted_sic
        risk_code
        risk_class
    """

    if model is None:
        model = load_model()

    result = predict_next_day_sic(
        input_df,
        model=model
    )

    result["risk_code"] = classify_sic_risk(
        result["predicted_sic"].values
    )

    labels = risk_labels()

    result["risk_class"] = (
        result["risk_code"].map(labels)
    )

    return result


def run_module1_spatial(
    input_df,
    yc_values,
    xc_values,
    model=None
):
    """
    Run Module 1 and return native spatial grid outputs.

    Returns:
        {
            "predicted_sic": 432x432 array,
            "risk_code": 432x432 array,
            "latitude": 432x432 array,
            "longitude": 432x432 array,
            "yc": 1D grid,
            "xc": 1D grid
        }
    """

    result = run_module1(
        input_df,
        model=model
    )

    return create_spatial_output(
        result,
        yc_values,
        xc_values
    )
