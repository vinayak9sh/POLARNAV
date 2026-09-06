
import numpy as np


# Prototype SIC screening thresholds.
# These are configurable and are NOT vessel-safety limits.
SIC_THRESHOLDS = {
    "low_max": 15.0,
    "moderate_max": 40.0,
    "high_max": 70.0,
}


RISK_LABELS = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
    3: "SEVERE",
}


def classify_sic_risk(sic_values):
    """
    Convert sea-ice concentration (%) to screening risk codes.

    Codes:
        0 = LOW
        1 = MODERATE
        2 = HIGH
        3 = SEVERE
       -1 = invalid / missing
    """

    sic_values = np.asarray(sic_values, dtype="float32")

    risk = np.full(
        sic_values.shape,
        -1,
        dtype=np.int8
    )

    valid = (
        np.isfinite(sic_values)
        & (sic_values >= 0)
        & (sic_values <= 100)
    )

    risk[
        valid & (sic_values < SIC_THRESHOLDS["low_max"])
    ] = 0

    risk[
        valid
        & (sic_values >= SIC_THRESHOLDS["low_max"])
        & (sic_values < SIC_THRESHOLDS["moderate_max"])
    ] = 1

    risk[
        valid
        & (sic_values >= SIC_THRESHOLDS["moderate_max"])
        & (sic_values < SIC_THRESHOLDS["high_max"])
    ] = 2

    risk[
        valid
        & (sic_values >= SIC_THRESHOLDS["high_max"])
        & (sic_values <= 100)
    ] = 3

    return risk


def risk_labels():
    """Return the mapping from risk code to label."""
    return RISK_LABELS.copy()
