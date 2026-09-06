
# Prototype vessel planning profiles.
#
# These weights represent routing preferences only.
# They are NOT operational or safety limits.

VESSEL_PROFILES = {

    "conservative": {
        "low": 1.0,
        "moderate": 10.0,
        "high": 50.0,
        "severe": 500.0
    },

    "standard": {
        "low": 1.0,
        "moderate": 5.0,
        "high": 20.0,
        "severe": 100.0
    },

    "ice_capable": {
        "low": 1.0,
        "moderate": 3.0,
        "high": 8.0,
        "severe": 30.0
    }
}


RISK_CODE_TO_NAME = {
    0: "low",
    1: "moderate",
    2: "high",
    3: "severe"
}


def get_vessel_profile(name):
    """
    Return a copy of a configured vessel planning profile.
    """

    if name not in VESSEL_PROFILES:
        raise ValueError(
            f"Unknown vessel profile: {name}. "
            f"Available profiles: "
            f"{list(VESSEL_PROFILES.keys())}"
        )

    return VESSEL_PROFILES[name].copy()


def create_vessel_cost_surface(
    risk_code,
    vessel_profile="standard"
):
    """
    Convert risk codes into a vessel-specific
    navigation cost surface.

    Code -1 remains blocked.
    """

    import numpy as np

    profile = get_vessel_profile(
        vessel_profile
    )

    risk_code = np.asarray(
        risk_code,
        dtype=np.int8
    )

    cost = np.full(
        risk_code.shape,
        np.inf,
        dtype=np.float32
    )

    for code, name in RISK_CODE_TO_NAME.items():

        cost[
            risk_code == code
        ] = profile[name]

    return cost
