from functools import lru_cache
# --------------------------------------------------------
# Module 1 — Sea ice
#
# Future dates are handled by the validated long-range
# forecast engine. Existing short-range dates continue
# using the original RF forecast.
# --------------------------------------------------------

from .module1_forecast import (
    forecast_sic,
    forecast_long_range_sic,
    get_long_range_forecast_window,
)
from .module1_risk import classify_sic_risk
from .vessel_profiles import create_vessel_cost_surface

from .module2_trajectory import (
    get_coverage_aware_predictions,
)

from .module2_risk import (
    create_weighted_iceberg_cost_surface,
)

from .module3_decision import (
    create_integrated_navigation_cost,
)


@lru_cache(maxsize=8)
def get_navigation_context(
    forecast_date: str,
    vessel_profile: str = "standard",
):
    """
    Build and cache the environmental navigation state
    for a forecast date and vessel profile.

    The expensive sea-ice and iceberg computations are
    performed only on the first request for each cache key.
    """

    # --------------------------------------------------------
    # Module 1 — Sea ice
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Module 1 — Sea ice
    #
    # Use the existing next-day RF forecast for dates that
    # belong to the short-range runtime. For future dates
    # beyond the latest observation, use the validated
    # long-range forecast engine.
    # --------------------------------------------------------

    window = get_long_range_forecast_window()

    latest_observation_date = window[
        "latest_observation_date"
    ]

    requested_date = forecast_date

    if requested_date > str(
        latest_observation_date.date()
    ):
        forecast_result = forecast_long_range_sic(
            requested_date
        )
    else:
        forecast_result = forecast_sic(
            requested_date
        )

    predicted_sic = (
        forecast_result["predicted_sic"]
    )

    risk_code = classify_sic_risk(
        predicted_sic
    )

    sea_ice_cost = create_vessel_cost_surface(
        risk_code,
        vessel_profile,
    )


    # --------------------------------------------------------
    # Module 2 — Icebergs
    # --------------------------------------------------------

    iceberg_coverage = (
        get_coverage_aware_predictions(
            forecast_date,
            max_persistence_days=3,
            persistence_hazard_weight=0.25,
        )
    )

    weighted_iceberg = (
        create_weighted_iceberg_cost_surface(
            forecast_result["latitude"],
            forecast_result["longitude"],
            iceberg_coverage["predictions"],
            influence_km=30.0,
            hard_avoid_km=5.0,
            max_penalty=500.0,
        )
    )

    iceberg_cost = (
        weighted_iceberg["iceberg_cost"]
    )


    # --------------------------------------------------------
    # Module 3 — Integrated navigation cost
    # --------------------------------------------------------

    navigation_cost = (
        create_integrated_navigation_cost(
            sea_ice_cost=sea_ice_cost,
            iceberg_cost=iceberg_cost,
            iceberg_weight=1.0,
        )
    )


    # --------------------------------------------------------
    # Spatial information required by Module 4
    # --------------------------------------------------------

    spatial_output = {
        "predicted_sic": predicted_sic,
        "risk_code": risk_code,
        "latitude": forecast_result["latitude"],
        "longitude": forecast_result["longitude"],
        "yc": forecast_result["yc"],
        "xc": forecast_result["xc"],
    }


    return {
        "forecast_result": forecast_result,
        "sea_ice_cost": sea_ice_cost,
        "iceberg_cost": iceberg_cost,
        "navigation_cost": navigation_cost,
        "spatial_output": spatial_output,
        "iceberg_coverage": iceberg_coverage,
    }


def clear_navigation_context_cache():
    """
    Clear cached navigation environments.
    Useful after changing model/data configuration.
    """

    get_navigation_context.cache_clear()