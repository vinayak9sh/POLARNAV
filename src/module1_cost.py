
import numpy as np


# Prototype navigation cost weights.
# These are tunable algorithmic parameters,
# not vessel-safety limits.
RISK_COSTS = {
    0: 1.0,        # LOW
    1: 5.0,        # MODERATE
    2: 20.0,       # HIGH
    3: 100.0,      # SEVERE
    -1: 1e6        # INVALID / unavailable
}


def create_navigation_cost(
    predicted_sic,
    risk_code=None
):
    """
    Convert predicted SIC into a navigation cost surface.

    If risk_code is supplied, risk-class costs are used.
    Otherwise a continuous SIC-based cost is generated.

    Returns:
        float32 numpy array
    """

    predicted_sic = np.asarray(
        predicted_sic,
        dtype="float32"
    )

    if risk_code is not None:

        risk_code = np.asarray(
            risk_code,
            dtype=np.int8
        )

        if predicted_sic.shape != risk_code.shape:
            raise ValueError(
                "predicted_sic and risk_code "
                "must have the same shape."
            )

        cost = np.full(
            predicted_sic.shape,
            RISK_COSTS[-1],
            dtype="float32"
        )

        for code, value in RISK_COSTS.items():

            if code == -1:
                continue

            cost[risk_code == code] = value

        return cost

    # Fallback continuous cost from SIC.
    valid = (
        np.isfinite(predicted_sic)
        & (predicted_sic >= 0)
        & (predicted_sic <= 100)
    )

    cost = np.full(
        predicted_sic.shape,
        RISK_COSTS[-1],
        dtype="float32"
    )

    # Smoothly increase cost with ice concentration.
    sic_fraction = predicted_sic[valid] / 100.0

    cost[valid] = (
        1.0
        + 20.0 * sic_fraction**2
    )

    return cost
