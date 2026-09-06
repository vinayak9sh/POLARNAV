
import numpy as np


# ============================================================
# Module 2 — Iceberg Proximity Risk
# ============================================================

# Prototype screening thresholds.
# These are NOT operational navigation safety limits.

DEFAULT_LOW_KM = 20.0
DEFAULT_MODERATE_KM = 10.0
DEFAULT_HIGH_KM = 5.0


def haversine_km_grid(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance in km.

    lat1/lon1 may be numpy arrays.
    lat2/lon2 may be scalars.
    """

    radius_km = 6371.0088

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad

    dlon = (
        lon2_rad
        - lon1_rad
    )

    dlon = (
        (dlon + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        radius_km
        * 2.0
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


def iceberg_distance_grid(
    grid_lat,
    grid_lon,
    iceberg_predictions
):
    """
    Return the minimum distance from each grid cell
    to any predicted iceberg.
    """

    grid_lat = np.asarray(grid_lat)
    grid_lon = np.asarray(grid_lon)

    min_distance = np.full(
        grid_lat.shape,
        np.inf,
        dtype=np.float32
    )

    for prediction in iceberg_predictions:

        lat = prediction.get(
            "predicted_latitude"
        )

        lon = prediction.get(
            "predicted_longitude"
        )

        if lat is None or lon is None:
            continue

        if not np.isfinite(lat) or not np.isfinite(lon):
            continue

        distance = haversine_km_grid(
            grid_lat,
            grid_lon,
            float(lat),
            float(lon)
        )

        min_distance = np.minimum(
            min_distance,
            distance.astype(np.float32)
        )

    return min_distance


def classify_iceberg_risk(
    distance_km,
    low_km=DEFAULT_LOW_KM,
    moderate_km=DEFAULT_MODERATE_KM,
    high_km=DEFAULT_HIGH_KM
):
    """
    Convert nearest-iceberg distance into a screening code.

    Codes:
        0 = LOW
        1 = MODERATE
        2 = HIGH
        3 = SEVERE
       -1 = invalid
    """

    distance_km = np.asarray(
        distance_km
    )

    risk = np.full(
        distance_km.shape,
        -1,
        dtype=np.int8
    )

    valid = np.isfinite(distance_km)

    risk[
        valid & (distance_km > low_km)
    ] = 0

    risk[
        valid
        & (distance_km <= low_km)
        & (distance_km > moderate_km)
    ] = 1

    risk[
        valid
        & (distance_km <= moderate_km)
        & (distance_km > high_km)
    ] = 2

    risk[
        valid
        & (distance_km <= high_km)
    ] = 3

    return risk


def create_iceberg_hazard_grid(
    grid_lat,
    grid_lon,
    iceberg_predictions,
    low_km=DEFAULT_LOW_KM,
    moderate_km=DEFAULT_MODERATE_KM,
    high_km=DEFAULT_HIGH_KM
):
    """
    Create iceberg proximity distance and risk grids.
    """

    if not iceberg_predictions:

        distance = np.full(
            np.asarray(grid_lat).shape,
            np.inf,
            dtype=np.float32
        )

    else:

        distance = iceberg_distance_grid(
            grid_lat,
            grid_lon,
            iceberg_predictions
        )

    risk = classify_iceberg_risk(
        distance,
        low_km=low_km,
        moderate_km=moderate_km,
        high_km=high_km
    )

    return {
        "nearest_distance_km": distance,
        "risk_code": risk
    }


def create_weighted_iceberg_cost_surface(
    grid_lat,
    grid_lon,
    predictions,
    influence_km=30.0,
    hard_avoid_km=5.0,
    max_penalty=500.0
):
    """
    Create a continuous iceberg navigation penalty
    using prediction confidence.

    Typical weights:
        ML prediction        -> 1.00
        Persistence estimate -> 0.25

    This is a planning cost, not an operational
    safety limit.
    """

    grid_lat = np.asarray(
        grid_lat,
        dtype=np.float32
    )

    grid_lon = np.asarray(
        grid_lon,
        dtype=np.float32
    )

    total_penalty = np.zeros(
        grid_lat.shape,
        dtype=np.float32
    )

    nearest_distance = np.full(
        grid_lat.shape,
        np.inf,
        dtype=np.float32
    )

    nearest_iceberg_weight = np.zeros(
        grid_lat.shape,
        dtype=np.float32
    )

    nearest_iceberg_id = np.full(
        grid_lat.shape,
        "",
        dtype=object
    )

    radius_km = 6371.0088

    lat1 = np.radians(grid_lat)
    lon1 = np.radians(grid_lon)

    for prediction in predictions:

        iceberg_id = str(
            prediction.get(
                "iceberg_id",
                "unknown"
            )
        )

        iceberg_lat = prediction.get(
            "predicted_latitude"
        )

        iceberg_lon = prediction.get(
            "predicted_longitude"
        )

        weight = float(
            prediction.get(
                "hazard_weight",
                1.0
            )
        )

        if (
            iceberg_lat is None
            or iceberg_lon is None
        ):
            continue

        if not (
            np.isfinite(iceberg_lat)
            and np.isfinite(iceberg_lon)
        ):
            continue

        if not np.isfinite(weight):
            continue

        weight = float(
            np.clip(weight, 0.0, 1.0)
        )

        # ----------------------------------------------------
        # Distance from every grid cell to this iceberg
        # ----------------------------------------------------

        lat2 = np.radians(
            float(iceberg_lat)
        )

        lon2 = np.radians(
            float(iceberg_lon)
        )

        dlat = lat2 - lat1

        dlon = (
            lon2
            - lon1
            + np.pi
        ) % (2.0 * np.pi) - np.pi

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2.0) ** 2
        )

        distance = (
            radius_km
            * 2.0
            * np.arcsin(
                np.sqrt(
                    np.clip(a, 0.0, 1.0)
                )
            )
        )

        # ----------------------------------------------------
        # Track nearest iceberg for explainability
        # ----------------------------------------------------

        closer = (
            distance < nearest_distance
        )

        nearest_distance[closer] = (
            distance[closer]
        )

        nearest_iceberg_weight[closer] = (
            weight
        )

        nearest_iceberg_id[closer] = (
            iceberg_id
        )

        # ----------------------------------------------------
        # Smooth proximity penalty
        # ----------------------------------------------------

        affected = (
            np.isfinite(distance)
            & (distance < influence_km)
            & (distance > hard_avoid_km)
        )

        if np.any(affected):

            proximity = (
                influence_km
                - distance[affected]
            ) / (
                influence_km
                - hard_avoid_km
            )

            contribution = (
                max_penalty
                * 0.20
                * proximity ** 2
                * weight
            )

            total_penalty[affected] += (
                contribution.astype(
                    np.float32
                )
            )

        # ----------------------------------------------------
        # Strong avoidance zone
        # ----------------------------------------------------

        hard_zone = (
            np.isfinite(distance)
            & (distance <= hard_avoid_km)
        )

        if np.any(hard_zone):

            hard_penalty = (
                max_penalty
                * weight
            )

            total_penalty[hard_zone] = np.maximum(
                total_penalty[hard_zone],
                np.float32(hard_penalty)
            )

    # Invalid geographic cells
    invalid = (
        ~np.isfinite(grid_lat)
        | ~np.isfinite(grid_lon)
    )

    total_penalty[invalid] = (
        np.float32(max_penalty)
    )

    return {
        "iceberg_cost": total_penalty,
        "nearest_distance_km": nearest_distance,
        "nearest_iceberg_weight": nearest_iceberg_weight,
        "nearest_iceberg_id": nearest_iceberg_id
    }
