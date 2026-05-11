from __future__ import annotations

from typing import Iterable

from utils.constants import HOURS_PER_DAY

from .config import ScenarioConfig

INFLEXIBLE_LOAD_SHAPE = [
    0.52, 0.50, 0.48, 0.47, 0.47, 0.50, 0.58, 0.72,
    0.82, 0.90, 0.96, 1.00, 0.98, 0.97, 0.95, 0.96,
    1.00, 0.98, 0.90, 0.82, 0.72, 0.64, 0.58, 0.54,
]

HVAC_LOAD_SHAPE = [
    0.18, 0.16, 0.15, 0.15, 0.15, 0.18, 0.28, 0.46,
    0.62, 0.78, 0.90, 0.96, 1.00, 0.98, 0.94, 0.96,
    1.00, 0.94, 0.76, 0.54, 0.34, 0.24, 0.20, 0.18,
]

SHIFTABLE_LOAD_SHAPE = [
    0.10, 0.08, 0.08, 0.08, 0.08, 0.10, 0.14, 0.30,
    0.52, 0.72, 0.88, 0.96, 1.00, 0.92, 0.82, 0.78,
    0.84, 0.90, 0.72, 0.46, 0.24, 0.16, 0.12, 0.10,
]

PV_SHAPE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.12, 0.26,
    0.45, 0.62, 0.78, 0.90, 1.00, 0.95, 0.82, 0.62,
    0.36, 0.12, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00,
]

EV_DEMAND_SHARE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08,
    0.10, 0.11, 0.11, 0.09, 0.07, 0.07, 0.08, 0.09,
    0.08, 0.07, 0.04, 0.02, 0.01, 0.01, 0.00, 0.00,
]

BUY_PRICE_RMB_PER_KWH = [
    0.45, 0.43, 0.42, 0.42, 0.43, 0.48, 0.56, 0.68,
    0.80, 0.84, 0.82, 0.76, 0.72, 0.70, 0.72, 0.78,
    0.88, 0.96, 1.02, 0.98, 0.86, 0.72, 0.60, 0.52,
]

GRID_CARBON_INTENSITY_KG_PER_KWH = [
    0.54, 0.53, 0.52, 0.52, 0.53, 0.55, 0.57, 0.60,
    0.58, 0.54, 0.50, 0.46, 0.44, 0.43, 0.44, 0.48,
    0.55, 0.62, 0.68, 0.70, 0.66, 0.61, 0.58, 0.56,
]

CARBON_PRICE_MULTIPLIER = [
    0.95, 0.95, 0.94, 0.94, 0.95, 0.98, 1.00, 1.04,
    1.06, 1.06, 1.03, 1.00, 0.98, 0.98, 1.00, 1.02,
    1.06, 1.10, 1.14, 1.12, 1.08, 1.04, 1.00, 0.98,
]


def _validate_24h_profile(values: Iterable[float], name: str) -> list[float]:
    profile = [round(float(value), 4) for value in values]
    if len(profile) != HOURS_PER_DAY:
        raise ValueError(f"{name} must contain exactly 24 hourly values.")
    return profile


def _scale_profile(shape: Iterable[float], peak_kw: float) -> list[float]:
    return [round(value * peak_kw, 3) for value in _validate_24h_profile(shape, "shape")]


def _normalize(weights: Iterable[float]) -> list[float]:
    values = _validate_24h_profile(weights, "weights")
    total = sum(values)
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return [value / total for value in values]


def build_base_profiles(config: ScenarioConfig) -> dict[str, object]:
    if config.horizon_hours != HOURS_PER_DAY:
        raise ValueError("The base profile library only supports a 24 h day-ahead case.")

    flexible_load_1, flexible_load_2 = config.flexible_loads
    ev_weights = _normalize(EV_DEMAND_SHARE)

    buy_price = _validate_24h_profile(BUY_PRICE_RMB_PER_KWH, "buy_price")
    sell_price = [round(value * 0.72, 4) for value in buy_price]
    carbon_price = [round(0.18 * multiplier, 4) for multiplier in _validate_24h_profile(CARBON_PRICE_MULTIPLIER, "carbon_price")]

    return {
        "hours": list(range(config.horizon_hours)),
        "inflexible_load_kw": _scale_profile(INFLEXIBLE_LOAD_SHAPE, config.inflexible_peak_kw),
        "flexible_loads_kw": {
            flexible_load_1.name: _scale_profile(HVAC_LOAD_SHAPE, flexible_load_1.baseline_peak_kw),
            flexible_load_2.name: _scale_profile(SHIFTABLE_LOAD_SHAPE, flexible_load_2.baseline_peak_kw),
        },
        "pv_available_kw": _scale_profile(PV_SHAPE, config.pv.rated_power_kw),
        "ev_energy_request_kwh": [
            round(weight * config.ev_cluster.daily_energy_kwh, 3)
            for weight in ev_weights
        ],
        "buy_price_rmb_per_kwh": buy_price,
        "sell_price_rmb_per_kwh": sell_price,
        "carbon_price_rmb_per_kg": carbon_price,
        "grid_carbon_intensity_kg_per_kwh": _validate_24h_profile(
            GRID_CARBON_INTENSITY_KG_PER_KWH,
            "grid_carbon_intensity",
        ),
    }
