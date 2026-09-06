
import heapq
import math
import numpy as np


def _heuristic(a, b):
    """Euclidean distance between two grid cells."""
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def find_least_cost_route(
    cost_surface,
    start,
    goal,
    blocked_cost=1e5
):
    """
    Find a least-cost route through a 2D navigation surface.

    Parameters
    ----------
    cost_surface : 2D numpy array
        Navigation cost for each grid cell.

    start : tuple
        (row, column)

    goal : tuple
        (row, column)

    blocked_cost : float
        Cells with cost >= this value are treated as blocked.

    Returns
    -------
    path : list of (row, column)
        Ordered route from start to goal.

    total_cost : float
        Accumulated route cost.
    """

    cost_surface = np.asarray(
        cost_surface,
        dtype="float32"
    )

    rows, cols = cost_surface.shape

    sr, sc = start
    gr, gc = goal

    if not (
        0 <= sr < rows
        and 0 <= sc < cols
        and 0 <= gr < rows
        and 0 <= gc < cols
    ):
        raise ValueError("Start or goal is outside the grid.")

    if (
        not np.isfinite(cost_surface[sr, sc])
        or cost_surface[sr, sc] >= blocked_cost
    ):
        raise ValueError("Start cell is blocked or invalid.")

    if (
        not np.isfinite(cost_surface[gr, gc])
        or cost_surface[gr, gc] >= blocked_cost
    ):
        raise ValueError("Goal cell is blocked or invalid.")

    # 8-connected neighbourhood
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    open_heap = []

    heapq.heappush(
        open_heap,
        (0.0, start)
    )

    g_score = {
        start: 0.0
    }

    came_from = {}

    while open_heap:

        _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct route
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()

            return path, g_score[goal]

        r, c = current

        for dr, dc in neighbours:

            nr = r + dr
            nc = c + dc

            if not (
                0 <= nr < rows
                and 0 <= nc < cols
            ):
                continue

            neighbour_cost = cost_surface[nr, nc]

            if (
                not np.isfinite(neighbour_cost)
                or neighbour_cost >= blocked_cost
            ):
                continue

            # Diagonal movement is slightly longer
            distance = math.sqrt(2) if dr != 0 and dc != 0 else 1.0

            step_cost = (
                0.5 * (
                    float(cost_surface[r, c])
                    + float(neighbour_cost)
                )
                * distance
            )

            tentative_g = (
                g_score[current]
                + step_cost
            )

            neighbour = (nr, nc)

            if (
                neighbour not in g_score
                or tentative_g < g_score[neighbour]
            ):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g

                f_score = (
                    tentative_g
                    + _heuristic(neighbour, goal)
                )

                heapq.heappush(
                    open_heap,
                    (f_score, neighbour)
                )

    raise RuntimeError(
        "No navigable route exists between start and goal."
    )



def build_route_result(
    route,
    total_cost,
    spatial_output,
    xc_values
):
    """
    Package an A* route into a backend-friendly result.

    Parameters
    ----------
    route : list of (row, col)
        Ordered A* route cells.

    total_cost : float
        Total navigation cost.

    spatial_output : dict
        Module 1 spatial output.

    xc_values : array
        Native projected X coordinates in km.

    Returns
    -------
    dict
        JSON-friendly route result.
    """

    import numpy as np

    route_rows = np.asarray(
        [p[0] for p in route],
        dtype=int
    )

    route_cols = np.asarray(
        [p[1] for p in route],
        dtype=int
    )

    route_x = np.asarray(xc_values)[route_cols]
    route_y = np.asarray(
        spatial_output["yc"]
    )[route_rows]

    route_lat = spatial_output["latitude"][
        route_rows,
        route_cols
    ]

    route_lon = spatial_output["longitude"][
        route_rows,
        route_cols
    ]

    route_sic = spatial_output["predicted_sic"][
        route_rows,
        route_cols
    ]

    route_risk = spatial_output["risk_code"][
        route_rows,
        route_cols
    ]

    # Physical route distance
    dx = np.diff(route_x)
    dy = np.diff(route_y)

    route_distance_km = float(
        np.sum(np.sqrt(dx**2 + dy**2))
    )

    # Straight-line distance
    straight_dx = route_x[-1] - route_x[0]
    straight_dy = route_y[-1] - route_y[0]

    straight_line_km = float(
        np.sqrt(
            straight_dx**2
            + straight_dy**2
        )
    )

    if straight_line_km > 0:
        detour_factor = (
            route_distance_km
            / straight_line_km
        )
    else:
        detour_factor = 1.0

    # Risk exposure
    risk_exposure = {}

    labels = {
        0: "LOW",
        1: "MODERATE",
        2: "HIGH",
        3: "SEVERE"
    }

    for code, label in labels.items():

        count = int(
            np.sum(route_risk == code)
        )

        risk_exposure[label] = {
            "cells": count,
            "percentage": round(
                100.0 * count / len(route),
                2
            )
        }

    valid_sic = route_sic[
        np.isfinite(route_sic)
    ]

    return {
        "status": "success",

        "start": {
            "latitude": float(route_lat[0]),
            "longitude": float(route_lon[0])
        },

        "destination": {
            "latitude": float(route_lat[-1]),
            "longitude": float(route_lon[-1])
        },

        "route": {
            "grid_cells": int(len(route)),
            "distance_km": round(
                route_distance_km,
                2
            ),
            "straight_line_distance_km": round(
                straight_line_km,
                2
            ),
            "detour_factor": round(
                detour_factor,
                3
            ),
            "extra_distance_km": round(
                route_distance_km
                - straight_line_km,
                2
            ),
            "total_navigation_cost": round(
                float(total_cost),
                2
            )
        },

        "ice_exposure": {
            "mean_predicted_sic_percent": round(
                float(np.nanmean(valid_sic)),
                2
            ),
            "max_predicted_sic_percent": round(
                float(np.nanmax(valid_sic)),
                2
            ),
            "risk_exposure": risk_exposure
        },

        "coordinates": [
            {
                "latitude": float(lat),
                "longitude": float(lon)
            }
            for lat, lon in zip(
                route_lat,
                route_lon
            )
        ],

        "geometry": {
            "type": "LineString",
            "coordinates": [
                [
                    float(lon),
                    float(lat)
                ]
                for lat, lon in zip(
                    route_lat,
                    route_lon
                )
            ]
        }
    }



def geographic_to_grid(
    latitude,
    longitude,
    spatial_output
):
    """
    Find the nearest valid Module 1 grid cell to a
    requested geographic coordinate.

    Parameters
    ----------
    latitude : float
    longitude : float
    spatial_output : dict
        Module 1 spatial output containing 2D latitude/longitude.

    Returns
    -------
    tuple
        (row, column)
    """

    import numpy as np

    lat_grid = np.asarray(
        spatial_output["latitude"],
        dtype="float32"
    )

    lon_grid = np.asarray(
        spatial_output["longitude"],
        dtype="float32"
    )

    valid = (
        np.isfinite(lat_grid)
        & np.isfinite(lon_grid)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid geographic grid cells available."
        )

    # Simple geographic nearest-neighbour search.
    # This is appropriate for the 432x432 MVP grid.
    distance_squared = (
        (lat_grid - float(latitude)) ** 2
        + (lon_grid - float(longitude)) ** 2
    )

    distance_squared[~valid] = np.inf

    flat_index = np.argmin(distance_squared)

    row, column = np.unravel_index(
        flat_index,
        lat_grid.shape
    )

    return int(row), int(column)



def plan_route(
    start_lat,
    start_lon,
    goal_lat,
    goal_lon,
    navigation_cost,
    spatial_output,
    blocked_cost=1e5
):
    """
    Plan an ice-aware least-cost route between two
    geographic coordinates.

    Parameters
    ----------
    start_lat, start_lon : float
        Starting geographic coordinate.

    goal_lat, goal_lon : float
        Destination geographic coordinate.

    navigation_cost : 2D array
        Navigation cost surface.

    spatial_output : dict
        Module 1 spatial output.

    blocked_cost : float
        Cost at or above this value is treated as blocked.

    Returns
    -------
    dict
        Backend-ready route result.
    """

    # Convert geographic coordinates to grid cells
    start_grid = geographic_to_grid(
        start_lat,
        start_lon,
        spatial_output
    )

    goal_grid = geographic_to_grid(
        goal_lat,
        goal_lon,
        spatial_output
    )

    # Calculate route
    route, total_cost = find_least_cost_route(
        navigation_cost,
        start_grid,
        goal_grid,
        blocked_cost=blocked_cost
    )

    # Package result
    result = build_route_result(
        route,
        total_cost,
        spatial_output,
        spatial_output["xc"]
    )

    # Preserve the user's requested endpoints
    result["requested_start"] = {
        "latitude": float(start_lat),
        "longitude": float(start_lon)
    }

    result["requested_destination"] = {
        "latitude": float(goal_lat),
        "longitude": float(goal_lon)
    }

    result["grid_start"] = {
        "row": int(start_grid[0]),
        "column": int(start_grid[1])
    }

    result["grid_goal"] = {
        "row": int(goal_grid[0]),
        "column": int(goal_grid[1])
    }

    return result
