
"""
Module 3 — Integrated Risk and Navigation Decision Engine

Combines:
    Module 1 vessel-specific sea-ice navigation cost
    Module 2 confidence-aware iceberg navigation cost

This module is a decision/fusion layer, not an ML model.
"""

from __future__ import annotations

import numpy as np


DEFAULT_ICEBERG_WEIGHT = 1.0


def create_integrated_navigation_cost(
    sea_ice_cost,
    iceberg_cost,
    iceberg_weight: float = DEFAULT_ICEBERG_WEIGHT
):
    """
    Combine sea-ice and iceberg navigation costs.

    Parameters
    ----------
    sea_ice_cost : array-like
        Vessel-specific sea-ice navigation cost surface.

    iceberg_cost : array-like
        Continuous iceberg navigation penalty surface.

    iceberg_weight : float
        Weight applied to the iceberg component.

    Returns
    -------
    np.ndarray
        Integrated navigation-cost surface.
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
            "sea_ice_cost and iceberg_cost "
            "must have identical shapes."
        )

    iceberg_weight = float(iceberg_weight)

    if not np.isfinite(iceberg_weight):
        raise ValueError(
            "iceberg_weight must be finite."
        )

    if iceberg_weight < 0:
        raise ValueError(
            "iceberg_weight cannot be negative."
        )

    integrated_cost = (
        sea_ice_cost
        + iceberg_weight * iceberg_cost
    ).astype(np.float32)

    # Preserve impassable cells from the sea-ice layer.
    invalid = (
        ~np.isfinite(sea_ice_cost)
        | ~np.isfinite(iceberg_cost)
    )

    integrated_cost[invalid] = np.inf

    return integrated_cost


def summarize_navigation_cost(
    sea_ice_cost,
    iceberg_cost,
    integrated_cost
):
    """
    Produce summary statistics for explainability.
    """

    sea_ice_cost = np.asarray(
        sea_ice_cost,
        dtype=np.float32
    )

    iceberg_cost = np.asarray(
        iceberg_cost,
        dtype=np.float32
    )

    integrated_cost = np.asarray(
        integrated_cost,
        dtype=np.float32
    )

    valid = np.isfinite(integrated_cost)

    if not np.any(valid):
        raise ValueError(
            "Integrated navigation cost contains "
            "no finite cells."
        )

    return {
        "valid_cells": int(np.sum(valid)),

        "sea_ice_cost": {
            "min": float(
                np.nanmin(sea_ice_cost[valid])
            ),
            "max": float(
                np.nanmax(sea_ice_cost[valid])
            ),
            "mean": float(
                np.nanmean(sea_ice_cost[valid])
            )
        },

        "iceberg_cost": {
            "min": float(
                np.nanmin(iceberg_cost[valid])
            ),
            "max": float(
                np.nanmax(iceberg_cost[valid])
            ),
            "mean": float(
                np.nanmean(iceberg_cost[valid])
            ),
            "affected_cells": int(
                np.sum(
                    iceberg_cost[valid] > 0
                )
            )
        },

        "integrated_cost": {
            "min": float(
                np.nanmin(integrated_cost[valid])
            ),
            "max": float(
                np.nanmax(integrated_cost[valid])
            ),
            "mean": float(
                np.nanmean(integrated_cost[valid])
            )
        }
    }


def explain_navigation_decision(
    sea_ice_cost,
    iceberg_cost,
    integrated_cost,
    iceberg_weight: float = DEFAULT_ICEBERG_WEIGHT
):
    """
    Explain the relative contribution of sea ice and
    iceberg hazards to the integrated navigation cost.
    """

    sea_ice_cost = np.asarray(
        sea_ice_cost,
        dtype=np.float32
    )

    iceberg_cost = np.asarray(
        iceberg_cost,
        dtype=np.float32
    )

    integrated_cost = np.asarray(
        integrated_cost,
        dtype=np.float32
    )

    valid = np.isfinite(integrated_cost)

    if not np.any(valid):
        raise ValueError(
            "No finite cells available for decision explanation."
        )

    sea_ice_values = sea_ice_cost[valid]
    iceberg_values = (
        iceberg_cost[valid] * float(iceberg_weight)
    )

    integrated_values = integrated_cost[valid]

    sea_ice_mean = float(
        np.nanmean(sea_ice_values)
    )

    iceberg_mean = float(
        np.nanmean(iceberg_values)
    )

    integrated_mean = float(
        np.nanmean(integrated_values)
    )

    total_component = (
        sea_ice_mean + iceberg_mean
    )

    if total_component > 0:
        sea_ice_share = (
            100.0 * sea_ice_mean / total_component
        )

        iceberg_share = (
            100.0 * iceberg_mean / total_component
        )
    else:
        sea_ice_share = 0.0
        iceberg_share = 0.0

    if sea_ice_share > iceberg_share:
        dominant_hazard = "sea_ice"

    elif iceberg_share > sea_ice_share:
        dominant_hazard = "iceberg"

    else:
        dominant_hazard = "balanced"

    return {
        "dominant_hazard": dominant_hazard,

        "mean_cost": {
            "sea_ice": sea_ice_mean,
            "iceberg": iceberg_mean,
            "integrated": integrated_mean
        },

        "contribution_percent": {
            "sea_ice": sea_ice_share,
            "iceberg": iceberg_share
        },

        "iceberg_weight": float(
            iceberg_weight
        )
    }


def explain_route_decision(
    route,
    sea_ice_cost,
    iceberg_cost,
    iceberg_weight: float = DEFAULT_ICEBERG_WEIGHT
):
    """
    Explain environmental cost contributions along
    the selected navigation route.

    Parameters
    ----------
    route : list of (row, col)
        Ordered route cells returned by the route optimizer.

    sea_ice_cost : array-like
        Vessel-specific sea-ice cost surface.

    iceberg_cost : array-like
        Weighted iceberg cost surface.

    iceberg_weight : float
        Additional multiplier applied to iceberg cost.

    Returns
    -------
    dict
        Route-level environmental contribution summary.
    """

    if route is None or len(route) == 0:
        raise ValueError(
            "Route is empty."
        )

    sea_ice_cost = np.asarray(
        sea_ice_cost,
        dtype=np.float32
    )

    iceberg_cost = np.asarray(
        iceberg_cost,
        dtype=np.float32
    )

    route_sea_ice = []
    route_iceberg = []

    for row, col in route:

        sea_value = sea_ice_cost[
            int(row),
            int(col)
        ]

        iceberg_value = iceberg_cost[
            int(row),
            int(col)
        ]

        if np.isfinite(sea_value):
            route_sea_ice.append(
                float(sea_value)
            )

        if np.isfinite(iceberg_value):
            route_iceberg.append(
                float(iceberg_value)
                * float(iceberg_weight)
            )

    if not route_sea_ice:
        raise ValueError(
            "No finite sea-ice costs along route."
        )

    sea_mean = float(
        np.mean(route_sea_ice)
    )

    sea_max = float(
        np.max(route_sea_ice)
    )

    iceberg_mean = (
        float(np.mean(route_iceberg))
        if route_iceberg
        else 0.0
    )

    iceberg_max = (
        float(np.max(route_iceberg))
        if route_iceberg
        else 0.0
    )

    combined_mean = (
        sea_mean + iceberg_mean
    )

    if combined_mean > 0:
        sea_share = (
            100.0
            * sea_mean
            / combined_mean
        )

        iceberg_share = (
            100.0
            * iceberg_mean
            / combined_mean
        )
    else:
        sea_share = 0.0
        iceberg_share = 0.0

    if sea_share > iceberg_share:
        dominant_hazard = "sea_ice"
    elif iceberg_share > sea_share:
        dominant_hazard = "iceberg"
    else:
        dominant_hazard = "balanced"

    return {
        "route_cells": int(len(route)),

        "dominant_hazard": dominant_hazard,

        "mean_cost": {
            "sea_ice": sea_mean,
            "iceberg": iceberg_mean,
            "combined": combined_mean
        },

        "maximum_cost": {
            "sea_ice": sea_max,
            "iceberg": iceberg_max
        },

        "iceberg_exposure": {
            "affected_route_cells": int(
                np.sum(
                    np.asarray(route_iceberg) > 0
                )
            ),
            "maximum_penalty": iceberg_max
        },

        "contribution_percent": {
            "sea_ice": sea_share,
            "iceberg": iceberg_share
        },

        "iceberg_affected_route_cells": int(
            np.sum(
                np.asarray(route_iceberg) > 0
            )
        )
    }
