
"""
Module 4 — Navigation and Dynamic Replanning

This module provides the higher-level navigation layer
around the existing A* route engine.

Responsibilities:
    1. Generate an initial vessel route.
    2. Replan from an updated vessel position.
    3. Compare old and new routes.
    4. Quantify route changes.

The underlying path search is delegated to module1_route.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .module1_route import plan_route
from .module1_multi_route import plan_multiple_routes


def plan_navigation_route(
    start_latitude: float,
    start_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    navigation_cost,
    spatial_output: Dict[str, Any],
    blocked_cost: float = 100000.0
) -> Dict[str, Any]:
    """
    Generate an initial navigation route.

    This is a thin decision-layer wrapper around the
    existing A* route planner.
    """

    result = plan_route(
        start_lat=start_latitude,
        start_lon=start_longitude,
        goal_lat=destination_latitude,
        goal_lon=destination_longitude,
        navigation_cost=navigation_cost,
        spatial_output=spatial_output,
        blocked_cost=blocked_cost
    )

    result["navigation_mode"] = "initial_route"

    return result


def replan_navigation_route(
    current_latitude: float,
    current_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    navigation_cost,
    spatial_output: Dict[str, Any],
    blocked_cost: float = 100000.0,
    previous_route: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recalculate the route from the vessel's updated
    position to the original destination according to the selected category.
    """

    try:
        routes = plan_multiple_routes(
            start_lat=current_latitude,
            start_lon=current_longitude,
            goal_lat=destination_latitude,
            goal_lon=destination_longitude,
            navigation_cost=navigation_cost,
            spatial_output=spatial_output,
            num_routes=3,
            blocked_cost=blocked_cost
        )
    except Exception:
        # Fallback to single route planning if multi-route generation fails
        routes = [
            plan_route(
                start_lat=current_latitude,
                start_lon=current_longitude,
                goal_lat=destination_latitude,
                goal_lon=destination_longitude,
                navigation_cost=navigation_cost,
                spatial_output=spatial_output,
                blocked_cost=blocked_cost
            )
        ]

    matched_route = None
    if category:
        cat_lower = category.lower()
        for r in routes:
            r_cat = r.get("category", "").lower()
            if cat_lower in r_cat or r_cat in cat_lower:
                matched_route = r
                break

    if matched_route is None:
        matched_route = routes[0]

    result = matched_route.copy()
    result["alternative_routes"] = routes
    result["navigation_mode"] = "replanned_route"

    if previous_route is not None:
        result["route_change"] = compare_routes(
            previous_route,
            result
        )
    else:
        result["route_change"] = {
            "comparison_available": False
        }

    return result


def compare_routes(
    previous_route: Dict[str, Any],
    new_route: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two route results.

    The comparison focuses on route geometry and
    navigation metrics.
    """

    previous_coordinates = (
        previous_route.get("coordinates", [])
    )

    new_coordinates = (
        new_route.get("coordinates", [])
    )

    previous_pairs = set()

    for point in previous_coordinates:

        if not isinstance(point, dict):
            continue

        lat = point.get("latitude")
        lon = point.get("longitude")

        if lat is None or lon is None:
            continue

        previous_pairs.add(
            (
                round(float(lat), 5),
                round(float(lon), 5)
            )
        )

    new_pairs = set()

    for point in new_coordinates:

        if not isinstance(point, dict):
            continue

        lat = point.get("latitude")
        lon = point.get("longitude")

        if lat is None or lon is None:
            continue

        new_pairs.add(
            (
                round(float(lat), 5),
                round(float(lon), 5)
            )
        )

    if not previous_pairs or not new_pairs:
        route_changed = None
        overlap_percent = None

    else:
        common = previous_pairs.intersection(
            new_pairs
        )

        smaller_route_size = min(
            len(previous_pairs),
            len(new_pairs)
        )

        overlap_percent = (
            100.0
            * len(common)
            / smaller_route_size
            if smaller_route_size > 0
            else 0.0
        )

        route_changed = (
            previous_pairs != new_pairs
        )

    previous_metrics = previous_route.get(
        "route",
        {}
    )

    new_metrics = new_route.get(
        "route",
        {}
    )

    previous_distance = float(
        previous_metrics.get(
            "distance_km",
            np.nan
        )
    )

    new_distance = float(
        new_metrics.get(
            "distance_km",
            np.nan
        )
    )

    previous_cost = float(
        previous_metrics.get(
            "total_navigation_cost",
            np.nan
        )
    )

    new_cost = float(
        new_metrics.get(
            "total_navigation_cost",
            np.nan
        )
    )

    if (
        np.isfinite(previous_distance)
        and np.isfinite(new_distance)
    ):
        distance_change_km = (
            new_distance
            - previous_distance
        )
    else:
        distance_change_km = None

    if (
        np.isfinite(previous_cost)
        and np.isfinite(new_cost)
    ):
        cost_change = (
            new_cost
            - previous_cost
        )
    else:
        cost_change = None

    return {
        "comparison_available": True,
        "route_changed": route_changed,
        "route_overlap_percent": (
            round(overlap_percent, 2)
            if overlap_percent is not None
            else None
        ),
        "previous_distance_km": (
            round(previous_distance, 2)
            if np.isfinite(previous_distance)
            else None
        ),
        "new_distance_km": (
            round(new_distance, 2)
            if np.isfinite(new_distance)
            else None
        ),
        "distance_change_km": (
            round(distance_change_km, 2)
            if distance_change_km is not None
            else None
        ),
        "previous_navigation_cost": (
            round(previous_cost, 4)
            if np.isfinite(previous_cost)
            else None
        ),
        "new_navigation_cost": (
            round(new_cost, 4)
            if np.isfinite(new_cost)
            else None
        ),
        "navigation_cost_change": (
            round(cost_change, 4)
            if cost_change is not None
            else None
        )
    }
