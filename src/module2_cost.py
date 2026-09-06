
import numpy as np


# ============================================================
# Module 2 — Continuous Iceberg Navigation Cost
# ============================================================

# These are prototype planning parameters.
# They are NOT operational safety limits.

DEFAULT_INFLUENCE_KM = 30.0
DEFAULT_HARD_AVOID_KM = 5.0

DEFAULT_MAX_PENALTY = 500.0


def create_iceberg_cost_surface(
    nearest_distance_km,
    influence_km=DEFAULT_INFLUENCE_KM,
    hard_avoid_km=DEFAULT_HARD_AVOID_KM,
    max_penalty=DEFAULT_MAX_PENALTY
):
    """
    Convert nearest-iceberg distance into a continuous
    navigation penalty.

    The penalty:
        - is 0 beyond influence_km
        - increases smoothly as distance decreases
        - becomes very large inside hard_avoid_km

    This is a planning cost, not a navigation safety limit.
    """

    distance = np.asarray(
        nearest_distance_km,
        dtype=np.float32
    )

    penalty = np.zeros_like(
        distance,
        dtype=np.float32
    )

    valid = np.isfinite(distance)

    # --------------------------------------------------------
    # Smooth proximity penalty
    # --------------------------------------------------------

    affected = (
        valid
        & (distance < influence_km)
        & (distance > hard_avoid_km)
    )

    # Normalized proximity:
    # 0 at influence boundary
    # 1 at hard-avoid boundary
    proximity = (
        influence_km - distance[affected]
    ) / (
        influence_km - hard_avoid_km
    )

    # Quadratic growth gives a gentle penalty far away
    # and much stronger penalty near the iceberg.
    penalty[affected] = (
        max_penalty
        * 0.20
        * proximity ** 2
    )

    # --------------------------------------------------------
    # Strong avoidance zone
    # --------------------------------------------------------

    hard_zone = (
        valid
        & (distance <= hard_avoid_km)
    )

    penalty[hard_zone] = max_penalty

    # --------------------------------------------------------
    # Invalid cells
    # --------------------------------------------------------

    penalty[~valid] = max_penalty

    return penalty


def combine_navigation_costs(
    sea_ice_cost,
    iceberg_cost,
    iceberg_weight=1.0
):
    """
    Combine Module 1 sea-ice navigation cost and
    Module 2 iceberg cost.

    The arrays must have identical shape.
    """

    sea_ice_cost = np.asarray(
        sea_ice_cost,
        dtype=np.float32
    )

    iceberg_cost = np.asarray(
        iceberg_cost,
        dtype=np.float32
    )

    if sea_ice_cost.shape != iceberg_cost.shape:
        raise ValueError(
            "Sea-ice and iceberg cost surfaces "
            "must have identical shapes."
        )

    return (
        sea_ice_cost
        + iceberg_weight * iceberg_cost
    ).astype(np.float32)
