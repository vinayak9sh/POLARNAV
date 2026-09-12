from pathlib import Path

from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .module4_navigation import (
    plan_navigation_route,
    replan_navigation_route,
)
from .module1_multi_route import (
    plan_multiple_routes,
)
from .module1_forecast import (
    forecast_sic,
    forecast_long_range_sic,
    get_long_range_forecast_window,
)
from .module1_risk import classify_sic_risk
from .module1_cost import create_navigation_cost
from .vessel_profiles import create_vessel_cost_surface

from .module2_trajectory import (
    predict_iceberg,
    list_icebergs,
    get_available_dates,
    get_coverage_aware_predictions,
)

from .module2_risk import (
    create_iceberg_hazard_grid,
    create_weighted_iceberg_cost_surface,
)

from .module3_decision import (
    create_integrated_navigation_cost,
    explain_route_decision,
)

from .navigation_context import (
    get_navigation_context,
)

app = FastAPI(
    title="Antarctic Navigation API",
    description="Prototype Module 1 sea-ice-aware route planning API",
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Existing vanilla frontend
        "http://127.0.0.1:5500",
        "http://localhost:5500",

        # New React/Vite frontend
        "http://127.0.0.1:5173",
        "http://localhost:5173"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    forecast_date: str
    vessel_profile: str = "standard"

    start_latitude: float
    start_longitude: float
    destination_latitude: float
    destination_longitude: float
    num_routes: int = 3


class ReplanRequest(BaseModel):
    forecast_date: str
    vessel_profile: str = "standard"

    current_latitude: float
    current_longitude: float
    destination_latitude: float
    destination_longitude: float
    category: str = "safest"


class ForecastRequest(BaseModel):
    forecast_date: str

class IcebergForecastRequest(BaseModel):
    iceberg_id: str
    forecast_date: str

# --------------------------------------------------
# Runtime data
# --------------------------------------------------

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

RUNTIME_PATH = (
    BASE_DIR
    / "models"
    / "module1_outputs"
    / "module1_navigation_runtime.npz"
)


def load_navigation_runtime():
    """
    Load the saved Module 1 spatial runtime artifact.
    """

    if not RUNTIME_PATH.exists():
        raise FileNotFoundError(
            f"Runtime artifact not found: {RUNTIME_PATH}"
        )

    data = np.load(
        RUNTIME_PATH,
        allow_pickle=False
    )

    spatial_output = {
        "predicted_sic": data["predicted_sic"],
        "risk_code": data["risk_code"],
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "yc": data["yc"],
        "xc": data["xc"],
    }

    navigation_cost = data["navigation_cost"]

    return navigation_cost, spatial_output


# Automatically load runtime data when API starts
try:
    NAVIGATION_COST, SPATIAL_OUTPUT = load_navigation_runtime()
    RUNTIME_LOADED = True
    RUNTIME_ERROR = None
except Exception as exc:
    NAVIGATION_COST = None
    SPATIAL_OUTPUT = None
    RUNTIME_LOADED = False
    RUNTIME_ERROR = str(exc)


# --------------------------------------------------
# Lightweight map grid
# --------------------------------------------------

# Downsample the native 432x432 grid by 4.
# This produces a 108x108 visualization grid and
# keeps browser/network load small.

GRID_STEP = 4

MAP_GRID = {
    "latitude": SPATIAL_OUTPUT["latitude"][::GRID_STEP, ::GRID_STEP],
    "longitude": SPATIAL_OUTPUT["longitude"][::GRID_STEP, ::GRID_STEP],
    "predicted_sic": SPATIAL_OUTPUT["predicted_sic"][::GRID_STEP, ::GRID_STEP],
    "risk_code": SPATIAL_OUTPUT["risk_code"][::GRID_STEP, ::GRID_STEP],
}


@app.get("/risk-grid")
def risk_grid():

    points = []

    lat = MAP_GRID["latitude"]
    lon = MAP_GRID["longitude"]
    sic = MAP_GRID["predicted_sic"]
    risk = MAP_GRID["risk_code"]

    rows, cols = lat.shape

    for i in range(rows):
        for j in range(cols):

            if (
                not np.isfinite(lat[i, j])
                or not np.isfinite(lon[i, j])
                or not np.isfinite(sic[i, j])
            ):
                continue

            code = int(risk[i, j])

            if code < 0:
                continue

            points.append({
                "latitude": float(lat[i, j]),
                "longitude": float(lon[i, j]),
                "sic": float(sic[i, j]),
                "risk_code": code
            })

    return {
        "status": "success",
        "grid_step": GRID_STEP,
        "points": points
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if RUNTIME_LOADED else "degraded",
        "module": "module1",
        "runtime_loaded": RUNTIME_LOADED
    }

# --------------------------------------------------
# Long-range forecast window
# --------------------------------------------------

@app.get("/forecast-window")
def forecast_window():
    """
    Return the currently available long-range SIC
    forecast window.
    """

    window = get_long_range_forecast_window()

    return {
        "latest_observation_date": (
            window["latest_observation_date"].strftime(
                "%Y-%m-%d"
            )
        ),
        "maximum_forecast_date": (
            window["maximum_forecast_date"].strftime(
                "%Y-%m-%d"
            )
        ),
        "max_forecast_months": (
            window["max_forecast_months"]
        ),
    }

# --------------------------------------------------
# Module 2 — Iceberg trajectory
# --------------------------------------------------

@app.get("/icebergs")
def icebergs():
    """
    Return all iceberg IDs available in the trajectory dataset.
    """

    try:
        ids = list_icebergs()

        return {
            "status": "success",
            "count": len(ids),
            "icebergs": ids
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


@app.get("/iceberg/{iceberg_id}")
def iceberg_info(iceberg_id: str):
    """
    Return availability information for one iceberg.
    """

    try:
        return {
            "status": "success",
            **get_available_dates(iceberg_id)
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )


@app.post("/iceberg-forecast")
def iceberg_forecast(
    request: IcebergForecastRequest
) -> Dict[str, Any]:
    """
    Predict the next-day position of an iceberg.
    """

    try:

        result = predict_iceberg(
            request.iceberg_id,
            request.forecast_date
        )

        return {
            "status": "success",
            "prediction": result
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

# --------------------------------------------------
# Module 2 — Iceberg risk grid
# --------------------------------------------------

@app.post("/iceberg-risk-grid")
def iceberg_risk_grid(
    request: ForecastRequest
) -> Dict[str, Any]:
    """
    Generate a coverage-aware, confidence-weighted iceberg
    navigation-risk grid for a forecast date.
    """

    try:

        # --------------------------------------------------
        # Module 1 spatial grid
        # --------------------------------------------------
        forecast_result = forecast_sic(
            request.forecast_date
        )

        lat = forecast_result["latitude"]
        lon = forecast_result["longitude"]

        # --------------------------------------------------
        # Module 2 — Coverage-aware predictions
        # --------------------------------------------------
        coverage_result = get_coverage_aware_predictions(
            request.forecast_date,
            max_persistence_days=3,
            persistence_hazard_weight=0.25
        )

        predictions = coverage_result["predictions"]
        coverage = coverage_result["coverage"]

        # --------------------------------------------------
        # Module 2 — Weighted continuous cost surface
        # --------------------------------------------------
        weighted = create_weighted_iceberg_cost_surface(
            lat,
            lon,
            predictions,
            influence_km=30.0,
            hard_avoid_km=5.0,
            max_penalty=500.0
        )

        iceberg_cost = weighted["iceberg_cost"]
        distance = weighted["nearest_distance_km"]
        weight = weighted["nearest_iceberg_weight"]
        iceberg_id = weighted["nearest_iceberg_id"]

        # --------------------------------------------------
        # Downsample for browser transmission
        # --------------------------------------------------
        step = 4

        lat_small = lat[::step, ::step]
        lon_small = lon[::step, ::step]
        cost_small = iceberg_cost[::step, ::step]
        distance_small = distance[::step, ::step]
        weight_small = weight[::step, ::step]
        id_small = iceberg_id[::step, ::step]

        # --------------------------------------------------
        # Restrict output to Antarctic region
        # --------------------------------------------------
        antarctic_mask = lat_small <= -55.0

        points = []

        rows, cols = lat_small.shape

        for i in range(rows):
            for j in range(cols):

                if (
                    not antarctic_mask[i, j]
                    or not np.isfinite(lat_small[i, j])
                    or not np.isfinite(lon_small[i, j])
                    or not np.isfinite(cost_small[i, j])
                    or not np.isfinite(distance_small[i, j])
                ):
                    continue

                nearest_id_value = id_small[i, j]

                if nearest_id_value is None:
                    nearest_id_value = ""

                points.append({
                    "latitude": float(
                        lat_small[i, j]
                    ),
                    "longitude": float(
                        lon_small[i, j]
                    ),
                    "iceberg_cost": float(
                        cost_small[i, j]
                    ),
                    "nearest_iceberg_distance_km": float(
                        distance_small[i, j]
                    ),
                    "nearest_iceberg_weight": float(
                        weight_small[i, j]
                    ),
                    "nearest_iceberg_id": str(
                        nearest_id_value
                    )
                })

        return {
            "status": "success",

            "forecast_date": str(
                forecast_result["forecast_date"]
            ),

            "prediction_date": str(
                forecast_result["prediction_date"]
            ),

            "grid_step": step,

            "iceberg_coverage": coverage,

            "cost_parameters": {
                "influence_radius_km": 30.0,
                "hard_avoid_radius_km": 5.0,
                "max_penalty": 500.0,
                "ml_hazard_weight": 1.0,
                "persistence_hazard_weight": 0.25
            },

            "predicted_icebergs": predictions,

            "points": points
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

# --------------------------------------------------
# Forecast
# --------------------------------------------------

@app.post("/forecast")
def forecast(request: ForecastRequest) -> Dict[str, Any]:
    """
    Generate a next-day SIC prediction for a selected date.
    """

    try:

        result = forecast_sic(
            request.forecast_date
        )

        predicted = result["predicted_sic"]

        valid = np.isfinite(predicted)

        return {
            "status": "success",
            "forecast_date": str(
                result["forecast_date"]
            ),
            "prediction_date": str(
                result["prediction_date"]
            ),
            "valid_cells": int(
                np.sum(valid)
            ),
            "predicted_sic": {
                "min_percent": round(
                    float(np.nanmin(predicted)),
                    2
                ),
                "max_percent": round(
                    float(np.nanmax(predicted)),
                    2
                ),
                "mean_percent": round(
                    float(np.nanmean(predicted)),
                    2
                )
            },
            "grid": {
                "rows": int(predicted.shape[0]),
                "columns": int(predicted.shape[1])
            }
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

# --------------------------------------------------
# Long-range SIC forecast
# --------------------------------------------------

@app.post("/forecast-long-range")
def forecast_long_range(
    request: ForecastRequest
) -> Dict[str, Any]:
    """
    Generate a validated 1–6 month SIC forecast.
    """

    try:
        result = forecast_long_range_sic(
            request.forecast_date
        )

        predicted = result["forecast_sic"]

        return {
            "status": "success",

            "forecast_date": str(
                result["forecast_date"].date()
            ),

            "latest_observation_date": str(
                result["latest_observation_date"].date()
            ),

            "horizon_months": (
                result["horizon_months"]
            ),

            "method": (
                result["method"]
            ),

            "alpha": (
                result["alpha"]
            ),

            "confidence": (
                result["confidence"]
            ),

            "uncertainty": {
                "typical_p75_percent_sic": (
                    result["uncertainty"]["typical_p75"]
                ),
                "conservative_p90_percent_sic": (
                    result["uncertainty"]["conservative_p90"]
                ),
            },

            "valid_cells": int(
                np.sum(
                    np.isfinite(predicted)
                )
            ),

            "predicted_sic": {
                "min_percent": round(
                    float(np.nanmin(predicted)),
                    2
                ),
                "max_percent": round(
                    float(np.nanmax(predicted)),
                    2
                ),
                "mean_percent": round(
                    float(np.nanmean(predicted)),
                    2
                )
            },

            "grid": {
                "rows": int(
                    predicted.shape[0]
                ),
                "columns": int(
                    predicted.shape[1]
                )
            }
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

# --------------------------------------------------
# Long-range SIC forecast grid
# --------------------------------------------------

@app.post("/forecast-long-range-grid")
def forecast_long_range_grid(
    request: ForecastRequest
) -> Dict[str, Any]:
    """
    Generate a long-range SIC forecast grid for a
    selected date, including empirical uncertainty
    and confidence metadata.
    """

    try:
        result = forecast_long_range_sic(
            request.forecast_date
        )

        predicted = result["forecast_sic"]

        # --------------------------------------------------
        # Risk classification
        # --------------------------------------------------

        risk_code = classify_sic_risk(
            predicted
        )

        # --------------------------------------------------
        # Downsample for browser transmission
        # --------------------------------------------------

        step = 4

        lat = result["latitude"][::step, ::step]
        lon = result["longitude"][::step, ::step]
        sic = predicted[::step, ::step]
        risk = risk_code[::step, ::step]

        rows, cols = lat.shape

        points = []

        for i in range(rows):
            for j in range(cols):

                if (
                    not np.isfinite(lat[i, j])
                    or not np.isfinite(lon[i, j])
                    or not np.isfinite(sic[i, j])
                    or risk[i, j] < 0
                ):
                    continue

                points.append({
                    "latitude": float(
                        lat[i, j]
                    ),
                    "longitude": float(
                        lon[i, j]
                    ),
                    "sic": float(
                        sic[i, j]
                    ),
                    "risk_code": int(
                        risk[i, j]
                    )
                })

        return {
            "status": "success",

            "forecast_date": str(
                result["forecast_date"].date()
            ),

            "latest_observation_date": str(
                result[
                    "latest_observation_date"
                ].date()
            ),

            "horizon_months": (
                result["horizon_months"]
            ),

            "method": (
                result["method"]
            ),

            "confidence": (
                result["confidence"]
            ),

            "uncertainty": {
                "typical_p75_percent_sic": (
                    result[
                        "uncertainty"
                    ]["typical_p75"]
                ),
                "conservative_p90_percent_sic": (
                    result[
                        "uncertainty"
                    ]["conservative_p90"]
                ),
            },

            "grid_step": step,

            "valid_cells": (
                result["valid_cells"]
            ),

            "predicted_sic": {
                "min_percent": round(
                    float(np.nanmin(predicted)),
                    2
                ),
                "max_percent": round(
                    float(np.nanmax(predicted)),
                    2
                ),
                "mean_percent": round(
                    float(np.nanmean(predicted)),
                    2
                )
            },

            "points": points
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

# --------------------------------------------------
# Forecast grid
# --------------------------------------------------

@app.post("/forecast-grid")
def forecast_grid(
    request: ForecastRequest
) -> Dict[str, Any]:
    """
    Generate a predicted SIC/risk visualization grid
    for a selected forecast date.
    """

    try:

        result = forecast_sic(
            request.forecast_date
        )

        predicted = result["predicted_sic"]

        risk_code = classify_sic_risk(
            predicted
        )

        # Downsample for browser visualization
        step = 4

        points = []

        lat = result["latitude"][::step, ::step]
        lon = result["longitude"][::step, ::step]
        sic = predicted[::step, ::step]
        risk = risk_code[::step, ::step]

        rows, cols = lat.shape

        for i in range(rows):
            for j in range(cols):

                if (
                    not np.isfinite(lat[i, j])
                    or not np.isfinite(lon[i, j])
                    or not np.isfinite(sic[i, j])
                    or risk[i, j] < 0
                ):
                    continue

                points.append({
                    "latitude": float(lat[i, j]),
                    "longitude": float(lon[i, j]),
                    "sic": float(sic[i, j]),
                    "risk_code": int(risk[i, j])
                })

        return {
            "status": "success",
            "forecast_date": str(
                result["forecast_date"]
            ),
            "prediction_date": str(
                result["prediction_date"]
            ),
            "grid_step": step,
            "valid_cells": int(
                np.sum(
                    np.isfinite(predicted)
                )
            ),
            "predicted_sic": {
                "min_percent": round(
                    float(np.nanmin(predicted)),
                    2
                ),
                "max_percent": round(
                    float(np.nanmax(predicted)),
                    2
                ),
                "mean_percent": round(
                    float(np.nanmean(predicted)),
                    2
                )
            },
            "points": points
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


# --------------------------------------------------
# Route
# --------------------------------------------------

@app.post("/route")
def route(request: RouteRequest) -> Dict[str, Any]:
    """
    Generate a sea-ice + iceberg-aware navigation route.
    """

    try:

        # --------------------------------------------------
        # Shared cached navigation environment
        # --------------------------------------------------
        navigation_context = get_navigation_context(
            request.forecast_date,
            request.vessel_profile
        )

        forecast_result = (
            navigation_context["forecast_result"]
        )

        predicted_sic = (
            forecast_result["predicted_sic"]
        )

        risk_code = classify_sic_risk(
            predicted_sic
        )

        sea_ice_navigation_cost = (
            navigation_context["sea_ice_cost"]
        )

        iceberg_navigation_cost = (
            navigation_context["iceberg_cost"]
        )

        navigation_cost = (
            navigation_context["navigation_cost"]
        )

        iceberg_coverage = (
            navigation_context["iceberg_coverage"]
        )

        # --------------------------------------------------
        # Spatial output expected by route engine
        # --------------------------------------------------
        spatial_output = {
            "predicted_sic": predicted_sic,
            "risk_code": risk_code,
            "latitude": forecast_result["latitude"],
            "longitude": forecast_result["longitude"],
            "yc": forecast_result["yc"],
            "xc": forecast_result["xc"],
        }

        # --------------------------------------------------
        # Multi-route Planning
        # --------------------------------------------------
        routes = plan_multiple_routes(
            start_lat=request.start_latitude,
            start_lon=request.start_longitude,
            goal_lat=request.destination_latitude,
            goal_lon=request.destination_longitude,
            navigation_cost=navigation_cost,
            spatial_output=spatial_output,
            num_routes=request.num_routes
        )

        for route_res in routes:
            # Module 3 — Explain decision along route
            route_coordinates = route_res.get("coordinates", [])
            route_grid_cells = []

            for point in route_coordinates:
                if not isinstance(point, dict):
                    continue

                route_latitude = point.get("latitude")
                route_longitude = point.get("longitude")

                if route_latitude is None or route_longitude is None:
                    continue

                cell_distance = (
                    np.abs(forecast_result["latitude"] - float(route_latitude))
                    + np.abs(forecast_result["longitude"] - float(route_longitude))
                )

                nearest_cell = np.unravel_index(
                    np.nanargmin(cell_distance),
                    cell_distance.shape
                )
                route_grid_cells.append(nearest_cell)

            if route_grid_cells:
                route_res["decision"] = explain_route_decision(
                    route_grid_cells,
                    sea_ice_navigation_cost,
                    iceberg_navigation_cost,
                    iceberg_weight=1.0
                )

            # Metadata
            route_res["forecast"] = {
                "forecast_date": str(forecast_result["forecast_date"]),
                "prediction_date": str(forecast_result["prediction_date"]),
                "horizon_months": forecast_result.get("horizon_months"),
                "method": forecast_result.get("method"),
                "confidence": forecast_result.get("confidence"),
                "uncertainty": forecast_result.get("uncertainty"),
            }

            route_res["vessel"] = {
                "profile": request.vessel_profile
            }

            coverage = iceberg_coverage["coverage"]
            route_res["iceberg"] = {
                "total_tracks": coverage["total_tracks"],
                "ml_predictions": coverage["ml_predictions"],
                "persistence_estimates": coverage["persistence_estimates"],
                "represented_tracks": coverage["represented_tracks"],
                "excluded_tracks": coverage["excluded_tracks"],
                "coverage_percent": coverage["coverage_percent"],
                "influence_radius_km": 30.0,
                "hard_avoid_radius_km": 5.0,
                "max_penalty": 500.0
            }

        # Primary route is at top-level for 100% backward compatibility
        primary_result = routes[0].copy()
        primary_result["alternative_routes"] = routes

        return primary_result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )



# --------------------------------------------------
# Module 4 — Dynamic route replanning
# --------------------------------------------------

@app.post("/replan")
def replan(request: ReplanRequest) -> Dict[str, Any]:
    """
    Recalculate a navigation route from the vessel's
    current position using an updated environmental state.
    """

    try:

                # --------------------------------------------------
        # Shared cached navigation environment
        # --------------------------------------------------
        navigation_context = get_navigation_context(
            request.forecast_date,
            request.vessel_profile
        )

        forecast_result = navigation_context["forecast_result"]

        predicted_sic = forecast_result["predicted_sic"]
        risk_code = classify_sic_risk(predicted_sic)

        sea_ice_navigation_cost = (
            navigation_context["sea_ice_cost"]
        )

        iceberg_navigation_cost = (
            navigation_context["iceberg_cost"]
        )

        navigation_cost = (
            navigation_context["navigation_cost"]
        )

        iceberg_coverage = (
            navigation_context["iceberg_coverage"]
        )

        # --------------------------------------------------
        # Spatial output expected by route engine
        # --------------------------------------------------
        spatial_output = navigation_context["spatial_output"]

        # --------------------------------------------------
        # Module 4 — Replan from current vessel position
        # --------------------------------------------------
        result = replan_navigation_route(
            current_latitude=request.current_latitude,
            current_longitude=request.current_longitude,
            destination_latitude=request.destination_latitude,
            destination_longitude=request.destination_longitude,
            navigation_cost=navigation_cost,
            spatial_output=spatial_output,
            category=request.category
        )

                # --------------------------------------------------
        # Module 3 — Explain replanned route
        # --------------------------------------------------

        route_coordinates = result.get(
            "coordinates",
            []
        )

        route_grid_cells = []

        for point in route_coordinates:

            if not isinstance(point, dict):
                continue

            route_latitude = point.get(
                "latitude"
            )

            route_longitude = point.get(
                "longitude"
            )

            if (
                route_latitude is None
                or route_longitude is None
            ):
                continue

            cell_distance = (
                np.abs(
                    forecast_result["latitude"]
                    - float(route_latitude)
                )
                +
                np.abs(
                    forecast_result["longitude"]
                    - float(route_longitude)
                )
            )

            nearest_cell = np.unravel_index(
                np.nanargmin(cell_distance),
                cell_distance.shape
            )

            route_grid_cells.append(
                nearest_cell
            )

        if route_grid_cells:

            result["decision"] = (
                explain_route_decision(
                    route_grid_cells,
                    sea_ice_navigation_cost,
                    iceberg_navigation_cost,
                    iceberg_weight=1.0
                )
            )

        # --------------------------------------------------
        # Forecast information
        # --------------------------------------------------
        result["forecast"] = {
            "forecast_date": str(
                forecast_result["forecast_date"]
            ),
            "prediction_date": str(
                forecast_result["prediction_date"]
            )
        }

        # --------------------------------------------------
        # Vessel information
        # --------------------------------------------------
        result["vessel"] = {
            "profile": request.vessel_profile
        }

        # --------------------------------------------------
        # Current vessel position
        # --------------------------------------------------
        result["current_position"] = {
            "latitude": request.current_latitude,
            "longitude": request.current_longitude
        }

        # --------------------------------------------------
        # Module 2 coverage information
        # --------------------------------------------------
        coverage = iceberg_coverage["coverage"]

        result["iceberg"] = {
            "total_tracks": coverage["total_tracks"],
            "ml_predictions": coverage["ml_predictions"],
            "persistence_estimates": coverage["persistence_estimates"],
            "represented_tracks": coverage["represented_tracks"],
            "excluded_tracks": coverage["excluded_tracks"],
            "coverage_percent": coverage["coverage_percent"],
            "influence_radius_km": 30.0,
            "hard_avoid_radius_km": 5.0,
            "max_penalty": 500.0
        }

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

