"""
Standalone SFOC-based fuel and cost estimation for POLARNAV.

This module is intentionally isolated:
- It does not modify the existing navigation system.
- It does not import POLARNAV navigation modules.
- It does not change A* routing.
- It does not require fuel data in the existing dataset.

The values in FuelProfile are configurable engineering assumptions
unless replaced with vessel-specific manufacturer/test data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuelProfile:
    """
    Vessel/fuel parameters used by the SFOC model.

    Parameters
    ----------
    sfoc_g_per_kwh:
        Specific Fuel Oil Consumption in grams per kilowatt-hour.
    reference_power_kw:
        Engine power corresponding to the reference operating point.
    reference_speed_knots:
        Vessel speed corresponding to reference_power_kw.
    fuel_price_usd_per_ton:
        Fuel price in USD per metric tonne.
    speed_power:
        Power-speed exponent. Default 3.0 is a simplified engineering model.
    """

    sfoc_g_per_kwh: float
    reference_power_kw: float
    reference_speed_knots: float
    fuel_price_usd_per_ton: float
    speed_power: float = 3.0


DEFAULT_FUEL_PROFILE = FuelProfile(
    sfoc_g_per_kwh=190.0,
    reference_power_kw=10_000.0,
    reference_speed_knots=14.0,
    fuel_price_usd_per_ton=700.0,
    speed_power=3.0,
)


def validate_profile(profile: FuelProfile) -> None:
    """Validate the vessel/fuel configuration."""
    if profile.sfoc_g_per_kwh <= 0:
        raise ValueError("SFOC must be greater than zero.")

    if profile.reference_power_kw <= 0:
        raise ValueError("Reference engine power must be greater than zero.")

    if profile.reference_speed_knots <= 0:
        raise ValueError("Reference speed must be greater than zero.")

    if profile.fuel_price_usd_per_ton < 0:
        raise ValueError("Fuel price cannot be negative.")

    if profile.speed_power <= 0:
        raise ValueError("Speed-power exponent must be greater than zero.")


def estimate_engine_power_kw(
    speed_knots: float,
    profile: FuelProfile = DEFAULT_FUEL_PROFILE,
    ice_penalty: float = 1.0,
) -> float:
    """
    Estimate propulsion power from vessel speed.

    Power is approximated using:

        P = P_ref * (v / v_ref)^n * ice_penalty

    Parameters
    ----------
    speed_knots:
        Vessel speed in knots.
    profile:
        Fuel/vessel configuration.
    ice_penalty:
        Multiplicative penalty for ice conditions.

    Returns
    -------
    float
        Estimated engine power in kW.
    """
    validate_profile(profile)

    if speed_knots <= 0:
        raise ValueError("Vessel speed must be greater than zero.")

    if ice_penalty <= 0:
        raise ValueError("Ice penalty must be greater than zero.")

    speed_ratio = speed_knots / profile.reference_speed_knots

    power_kw = (
        profile.reference_power_kw
        * speed_ratio ** profile.speed_power
    )

    return power_kw * ice_penalty


def distance_km_to_nm(distance_km: float) -> float:
    """Convert kilometres to nautical miles."""
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    return distance_km / 1.852


def calculate_travel_time_hours(
    distance_km: float,
    speed_knots: float,
) -> float:
    """
    Calculate travel time from distance and vessel speed.
    """
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if speed_knots <= 0:
        raise ValueError("Vessel speed must be greater than zero.")

    distance_nm = distance_km_to_nm(distance_km)

    return distance_nm / speed_knots


def calculate_fuel_tons(
    distance_km: float,
    speed_knots: float,
    profile: FuelProfile = DEFAULT_FUEL_PROFILE,
    ice_penalty: float = 1.0,
) -> float:
    """
    Estimate fuel consumption in metric tonnes.

    SFOC relation:

        fuel_g = SFOC * power_kW * time_h

    Then:

        fuel_tons = fuel_g / 1,000,000
    """
    validate_profile(profile)

    travel_hours = calculate_travel_time_hours(
        distance_km,
        speed_knots,
    )

    power_kw = estimate_engine_power_kw(
        speed_knots,
        profile,
        ice_penalty,
    )

    fuel_grams = (
        profile.sfoc_g_per_kwh
        * power_kw
        * travel_hours
    )

    return fuel_grams / 1_000_000.0


def calculate_fuel_cost(
    fuel_tons: float,
    profile: FuelProfile = DEFAULT_FUEL_PROFILE,
) -> float:
    """Convert fuel consumption into USD cost."""
    validate_profile(profile)

    if fuel_tons < 0:
        raise ValueError("Fuel consumption cannot be negative.")

    return fuel_tons * profile.fuel_price_usd_per_ton


def estimate_voyage(
    distance_km: float,
    speed_knots: float,
    profile: FuelProfile = DEFAULT_FUEL_PROFILE,
    ice_penalty: float = 1.0,
) -> dict:
    """
    Calculate the main fuel/cost metrics for a voyage.
    """
    travel_hours = calculate_travel_time_hours(
        distance_km,
        speed_knots,
    )

    engine_power_kw = estimate_engine_power_kw(
        speed_knots,
        profile,
        ice_penalty,
    )

    fuel_tons = calculate_fuel_tons(
        distance_km,
        speed_knots,
        profile,
        ice_penalty,
    )

    fuel_cost_usd = calculate_fuel_cost(
        fuel_tons,
        profile,
    )

    return {
        "distance_km": round(distance_km, 2),
        "speed_knots": round(speed_knots, 2),
        "travel_time_hours": round(travel_hours, 2),
        "estimated_engine_power_kw": round(engine_power_kw, 2),
        "sfoc_g_per_kwh": profile.sfoc_g_per_kwh,
        "ice_penalty": round(ice_penalty, 3),
        "fuel_used_tons": round(fuel_tons, 3),
        "fuel_price_usd_per_ton": round(
            profile.fuel_price_usd_per_ton,
            2,
        ),
        "fuel_cost_usd": round(fuel_cost_usd, 2),
    }


if __name__ == "__main__":
    # Simple standalone test.
    result = estimate_voyage(
        distance_km=500.0,
        speed_knots=12.0,
        profile=DEFAULT_FUEL_PROFILE,
        ice_penalty=1.0,
    )

    print("POLARNAV SFOC Fuel Test")
    print("-----------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
