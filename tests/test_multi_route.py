"""
Tests for Multiple Alternative Routes module (src/module1_multi_route.py)
"""

import numpy as np
import pytest

from src.module1_multi_route import (
    apply_corridor_penalty,
    calculate_route_overlap,
    plan_multiple_routes,
    recalculate_unpenalized_route_cost,
)


def test_calculate_route_overlap():
    route_a = [(0, 0), (0, 1), (0, 2), (0, 3)]
    route_b = [(0, 0), (0, 1), (1, 2), (1, 3)]
    route_c = [(5, 5), (5, 6), (5, 7)]

    assert calculate_route_overlap(route_a, route_a) == 1.0
    assert calculate_route_overlap(route_a, route_b) == 0.5
    assert calculate_route_overlap(route_a, route_c) == 0.0
    assert calculate_route_overlap([], route_a) == 0.0


def test_recalculate_unpenalized_route_cost():
    cost_grid = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    route = [(0, 0), (0, 1), (1, 1)]
    # Step 1: (0,0) -> (0,1): distance=1.0, cost = 0.5*(1.0 + 2.0)*1.0 = 1.5
    # Step 2: (0,1) -> (1,1): distance=1.0, cost = 0.5*(2.0 + 5.0)*1.0 = 3.5
    # Total = 5.0
    total = recalculate_unpenalized_route_cost(route, cost_grid)
    assert pytest.approx(total, 0.01) == 5.0


def test_apply_corridor_penalty():
    cost_grid = np.ones((10, 10), dtype=np.float32)
    route = [(5, 5)]

    penalized = apply_corridor_penalty(
        cost_grid,
        route,
        penalty_multiplier=3.0,
        radius_cells=2
    )

    # Center cell (5,5) should have maximum multiplier 3.0
    assert pytest.approx(penalized[5, 5], 0.01) == 3.0
    # Cell far away (0,0) should remain 1.0
    assert penalized[0, 0] == 1.0
    # Cell inside corridor radius (5,6) should be > 1.0 and <= 3.0
    assert 1.0 < penalized[5, 6] <= 3.0


def test_plan_multiple_routes_synthetic():
    # 20x20 uniform cost surface
    cost_surface = np.ones((20, 20), dtype=np.float32)
    # Put a barrier of high cost in middle to encourage distinct paths
    cost_surface[8:12, 8:12] = 50.0

    lats = np.linspace(-60.0, -50.0, 20)
    lons = np.linspace(0.0, 10.0, 20)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    spatial_output = {
        "predicted_sic": np.zeros((20, 20)),
        "risk_code": np.zeros((20, 20)),
        "latitude": lat_grid,
        "longitude": lon_grid,
        "yc": lats,
        "xc": lons
    }

    routes = plan_multiple_routes(
        start_lat=-60.0,
        start_lon=0.0,
        goal_lat=-50.0,
        goal_lon=10.0,
        navigation_cost=cost_surface,
        spatial_output=spatial_output,
        num_routes=3,
        max_overlap_ratio=0.85
    )

    assert len(routes) >= 1
    assert routes[0]["is_primary"] is True
    assert routes[0]["route_id"] == "primary"
    
    for r in routes:
        assert "geometry" in r
        assert "route" in r
        assert r["route"]["distance_km"] > 0
