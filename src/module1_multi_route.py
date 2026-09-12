"""
Module 1 — Multiple Alternative Route Generator

Provides multi-route planning capabilities for POLARNAV.
Generates spatially distinct alternative navigation routes between the same
start and destination using Penalty-Based Iterative A*.

This module is isolated and reuses the core A* solver in module1_route
without altering existing single-route functionality or ML models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .module1_route import (
    build_route_result,
    find_least_cost_route,
    geographic_to_grid,
    plan_route,
)


def calculate_route_overlap(
    route_a: List[Tuple[int, int]],
    route_b: List[Tuple[int, int]]
) -> float:
    """
    Calculate the spatial cell overlap ratio between two routes.

    Parameters
    ----------
    route_a : list of (row, col)
    route_b : list of (row, col)

    Returns
    -------
    float
        Overlap ratio between 0.0 (completely distinct) and 1.0 (identical).
    """
    if not route_a or not route_b:
        return 0.0

    set_a = set(route_a)
    set_b = set(route_b)

    common = set_a.intersection(set_b)
    smaller_length = min(len(set_a), len(set_b))

    if smaller_length == 0:
        return 0.0

    return len(common) / float(smaller_length)


def recalculate_unpenalized_route_cost(
    route: List[Tuple[int, int]],
    original_cost_surface: np.ndarray,
    blocked_cost: float = 1e5
) -> float:
    """
    Recalculate the true, unpenalized total cost of a path on the original cost surface.

    Parameters
    ----------
    route : list of (row, col)
    original_cost_surface : 2D numpy array
    blocked_cost : float

    Returns
    -------
    float
        Unpenalized total navigation cost.
    """
    total_cost = 0.0
    for i in range(len(route) - 1):
        r1, c1 = route[i]
        r2, c2 = route[i + 1]

        cost1 = float(original_cost_surface[r1, c1])
        cost2 = float(original_cost_surface[r2, c2])

        dr = r2 - r1
        dc = c2 - c1
        distance = math.sqrt(2) if dr != 0 and dc != 0 else 1.0

        step_cost = 0.5 * (cost1 + cost2) * distance
        total_cost += step_cost

    return total_cost


def apply_corridor_penalty(
    cost_surface: np.ndarray,
    route: List[Tuple[int, int]],
    penalty_multiplier: float = 3.0,
    radius_cells: int = 6,
    blocked_cost: float = 1e5
) -> np.ndarray:
    """
    Apply a spatial corridor penalty to a cost surface around existing route cells.

    Parameters
    ----------
    cost_surface : 2D numpy array
        Original cost surface.
    route : list of (row, col)
        Ordered grid cells of previous route.
    penalty_multiplier : float
        Multiplier factor applied to cell costs in the corridor.
    radius_cells : int
        Grid radius in cells around the route to penalize.
    blocked_cost : float
        Threshold for blocked cells.

    Returns
    -------
    np.ndarray
        Penalized cost surface.
    """
    penalized = np.copy(cost_surface)
    rows, cols = penalized.shape

    # Collect unique cells to avoid double-penalizing overlaps in same route
    visited_penalty = {}

    for pr, pc in route:
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr = pr + dr
                nc = pc + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    cell_cost = penalized[nr, nc]
                    if np.isfinite(cell_cost) and cell_cost < blocked_cost:
                        dist = math.hypot(dr, dc)
                        if dist <= radius_cells:
                            # Linear decay factor from center of corridor
                            weight = 1.0 - (dist / (radius_cells + 1.0))
                            penalty_factor = 1.0 + (penalty_multiplier - 1.0) * weight
                            
                            key = (nr, nc)
                            if key not in visited_penalty or penalty_factor > visited_penalty[key]:
                                visited_penalty[key] = penalty_factor

    for (nr, nc), factor in visited_penalty.items():
        penalized[nr, nc] *= factor

    return penalized


def plan_multiple_routes(
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    navigation_cost: np.ndarray,
    spatial_output: Dict[str, Any],
    num_routes: int = 3,
    max_overlap_ratio: float = 0.85,
    blocked_cost: float = 1e5
) -> List[Dict[str, Any]]:
    """
    Plan multiple viable, spatially distinct navigation routes between
    start and goal coordinates.

    Parameters
    ----------
    start_lat, start_lon : float
        Starting geographic coordinates.
    goal_lat, goal_lon : float
        Destination geographic coordinates.
    navigation_cost : 2D numpy array
        Integrated navigation cost surface.
    spatial_output : dict
        Spatial grid metadata.
    num_routes : int
        Number of alternative routes to request (default 3).
    max_overlap_ratio : float
        Maximum allowable cell overlap ratio between alternative routes (default 0.85).
    blocked_cost : float
        Threshold above which cells are treated as impassable.

    Returns
    -------
    list of dict
        Ordered list of route results (Primary/Optimal route first, followed by Alternatives).
    """
    # 1. Primary Route (Optimal)
    primary_result = plan_route(
        start_lat=start_lat,
        start_lon=start_lon,
        goal_lat=goal_lat,
        goal_lon=goal_lon,
        navigation_cost=navigation_cost,
        spatial_output=spatial_output,
        blocked_cost=blocked_cost
    )

    primary_result["route_id"] = "primary"
    primary_result["route_name"] = "Primary Route (Optimal)"
    primary_result["is_primary"] = True

    routes = [primary_result]

    if num_routes <= 1:
        return routes

    # Convert coordinates to grid cells for pathfinding
    start_grid = geographic_to_grid(start_lat, start_lon, spatial_output)
    goal_grid = geographic_to_grid(goal_lat, goal_lon, spatial_output)

    # Extract primary route grid cell sequence
    route_grid_cells_list = []
    
    # We can reconstruct grid cells from geographic to grid or store them
    # For accuracy, extract grid cells from geographic coordinates
    primary_coords = primary_result.get("coordinates", [])
    primary_grid_cells = [
        geographic_to_grid(pt["latitude"], pt["longitude"], spatial_output)
        for pt in primary_coords
    ]
    route_grid_cells_list.append(primary_grid_cells)

    current_cost_surface = np.copy(navigation_cost)

    penalty_multipliers = [3.0, 6.0, 12.0, 25.0]
    radii = [6, 10, 14, 18]

    attempts = 0
    max_attempts = len(penalty_multipliers) * 2

    while len(routes) < num_routes and attempts < max_attempts:
        mult = penalty_multipliers[attempts % len(penalty_multipliers)]
        rad = radii[attempts % len(radii)]
        attempts += 1

        # Apply corridor penalties for all previously accepted routes
        working_surface = np.copy(navigation_cost)
        for prev_route_cells in route_grid_cells_list:
            working_surface = apply_corridor_penalty(
                working_surface,
                prev_route_cells,
                penalty_multiplier=mult,
                radius_cells=rad,
                blocked_cost=blocked_cost
            )

        try:
            alt_grid_cells, _ = find_least_cost_route(
                working_surface,
                start_grid,
                goal_grid,
                blocked_cost=blocked_cost
            )
        except RuntimeError:
            # No alternative route could be found under this penalty constraint
            continue

        # Check overlap against all accepted routes
        is_distinct = True
        for prev_route_cells in route_grid_cells_list:
            overlap = calculate_route_overlap(alt_grid_cells, prev_route_cells)
            if overlap >= max_overlap_ratio:
                is_distinct = False
                break

        if not is_distinct:
            continue

        # Recalculate true unpenalized navigation cost on original surface
        unpenalized_cost = recalculate_unpenalized_route_cost(
            alt_grid_cells,
            navigation_cost,
            blocked_cost=blocked_cost
        )

        alt_result = build_route_result(
            alt_grid_cells,
            unpenalized_cost,
            spatial_output,
            spatial_output["xc"]
        )

        alt_idx = len(routes)
        alt_result["route_id"] = f"alternative_{alt_idx}"
        alt_result["route_name"] = f"Alternative Route {alt_idx}"
        alt_result["is_primary"] = False

        alt_result["requested_start"] = {
            "latitude": float(start_lat),
            "longitude": float(start_lon)
        }
        alt_result["requested_destination"] = {
            "latitude": float(goal_lat),
            "longitude": float(goal_lon)
        }
        alt_result["grid_start"] = {
            "row": int(start_grid[0]),
            "column": int(start_grid[1])
        }
        alt_result["grid_goal"] = {
            "row": int(goal_grid[0]),
            "column": int(goal_grid[1])
        }

        routes.append(alt_result)
        route_grid_cells_list.append(alt_grid_cells)

    return routes
