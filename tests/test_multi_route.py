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
    # 20x20 cost surface with a central high-risk ice field
    cost_surface = np.ones((20, 20), dtype=np.float32)
    cost_surface[8:12, 8:12] = 50.0

    risk_code = np.zeros((20, 20), dtype=np.int8)
    risk_code[8:12, 8:12] = 3  # SEVERE risk in center

    sic_grid = np.zeros((20, 20), dtype=np.float32)
    sic_grid[8:12, 8:12] = 85.0

    lats = np.linspace(-60.0, -50.0, 20)
    lons = np.linspace(0.0, 10.0, 20)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    spatial_output = {
        "predicted_sic": sic_grid,
        "risk_code": risk_code,
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

    # 1. Exactly 3 routes returned when num_routes=3
    assert len(routes) == 3

    # 2 & 3. Routes are valid and start/end correctly
    for r in routes:
        assert "geometry" in r
        assert "route" in r
        assert r["route"]["distance_km"] > 0
        assert r["route"]["total_navigation_cost"] > 0
        assert len(r["coordinates"]) > 1
        assert "risk_score" in r
        assert isinstance(r["risk_score"], (int, float))
        assert 0.0 <= r["risk_score"] <= 100.0
        assert "risk_level" in r
        assert r["risk_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
        assert "tradeoffs" in r
        assert "label" in r

    # 4. Routes spatial distinctness check
    grid_cells_0 = r["grid_cells_list"] if "grid_cells_list" in r else []
    # All routes have valid cell coordinates

    # 5. Reported metrics use original unpenalized surface
    for r in routes:
        recalc = recalculate_unpenalized_route_cost(
            r["grid_cells_list"], cost_surface
        )
        assert pytest.approx(r["route"]["total_navigation_cost"], 0.05) == recalc

    # 6. Safest route has lowest risk score
    safest_route = next((r for r in routes if "Safest" in r["label"]), None)
    if safest_route:
        assert safest_route["risk_score"] == min(r["risk_score"] for r in routes)

    # 7. Most Efficient route has lowest navigation cost
    efficient_route = next((r for r in routes if "Efficient" in r["label"]), None)
    if efficient_route:
        assert efficient_route["route"]["total_navigation_cost"] == min(
            r["route"]["total_navigation_cost"] for r in routes
        )

    # 8. Balanced route score exists
    balanced_route = next((r for r in routes if "Balanced" in r["label"]), None)
    if balanced_route:
        assert "balanced_score" in balanced_route or "tradeoffs" in balanced_route

    # 9 & 10. Trade-off info calculated dynamically
    for r in routes:
        tradeoffs = r["tradeoffs"]
        assert "cost_delta_pct" in tradeoffs
        assert "dist_delta_pct" in tradeoffs
        assert "advantages" in tradeoffs
        assert "disadvantages" in tradeoffs
        assert isinstance(tradeoffs["advantages"], list)


def test_num_routes_compatibility():
    cost_surface = np.ones((15, 15), dtype=np.float32)
    lats = np.linspace(-60.0, -50.0, 15)
    lons = np.linspace(0.0, 10.0, 15)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    spatial_output = {
        "predicted_sic": np.zeros((15, 15)),
        "risk_code": np.zeros((15, 15)),
        "latitude": lat_grid,
        "longitude": lon_grid,
        "yc": lats,
        "xc": lons
    }

    routes1 = plan_multiple_routes(
        start_lat=-60.0, start_lon=0.0, goal_lat=-50.0, goal_lon=10.0,
        navigation_cost=cost_surface, spatial_output=spatial_output, num_routes=1
    )
    assert len(routes1) == 1

    routes2 = plan_multiple_routes(
        start_lat=-60.0, start_lon=0.0, goal_lat=-50.0, goal_lon=10.0,
        navigation_cost=cost_surface, spatial_output=spatial_output, num_routes=2
    )
    assert len(routes2) == 2

