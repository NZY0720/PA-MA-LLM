from __future__ import annotations

import math
from typing import Iterable

from .config import ScenarioConfig

DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24
HOURS_PER_WEEK = DAYS_PER_WEEK * HOURS_PER_DAY
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _validate_profile(values: Iterable[float], name: str) -> list[float]:
    profile = [round(float(value), 4) for value in values]
    if len(profile) != HOURS_PER_DAY:
        raise ValueError(f"{name} must contain exactly {HOURS_PER_DAY} hourly values.")
    return profile


def _normalize(weights: Iterable[float]) -> list[float]:
    values = _validate_profile(weights, "weights")
    total = sum(values)
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return [value / total for value in values]


WEEKDAY_INFLEXIBLE = _validate_profile(
    [
        0.42,
        0.40,
        0.39,
        0.39,
        0.40,
        0.45,
        0.56,
        0.70,
        0.82,
        0.90,
        0.96,
        1.00,
        0.99,
        0.98,
        0.97,
        0.98,
        1.00,
        0.98,
        0.90,
        0.78,
        0.66,
        0.56,
        0.49,
        0.45,
    ],
    "weekday_inflexible",
)

WEEKEND_INFLEXIBLE = _validate_profile(
    [
        0.36,
        0.35,
        0.34,
        0.34,
        0.35,
        0.38,
        0.42,
        0.48,
        0.54,
        0.60,
        0.68,
        0.74,
        0.76,
        0.75,
        0.73,
        0.74,
        0.77,
        0.75,
        0.68,
        0.60,
        0.52,
        0.46,
        0.42,
        0.39,
    ],
    "weekend_inflexible",
)

WEEKDAY_HVAC = _validate_profile(
    [
        0.16,
        0.15,
        0.14,
        0.14,
        0.15,
        0.18,
        0.28,
        0.44,
        0.60,
        0.76,
        0.88,
        0.95,
        1.00,
        0.98,
        0.95,
        0.96,
        1.00,
        0.94,
        0.78,
        0.58,
        0.40,
        0.28,
        0.22,
        0.18,
    ],
    "weekday_hvac",
)

WEEKEND_HVAC = _validate_profile(
    [
        0.14,
        0.13,
        0.12,
        0.12,
        0.13,
        0.16,
        0.22,
        0.30,
        0.42,
        0.54,
        0.66,
        0.76,
        0.82,
        0.80,
        0.77,
        0.76,
        0.78,
        0.74,
        0.62,
        0.48,
        0.34,
        0.24,
        0.18,
        0.15,
    ],
    "weekend_hvac",
)

WEEKDAY_SERVICE = _validate_profile(
    [
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.10,
        0.16,
        0.28,
        0.48,
        0.68,
        0.84,
        0.94,
        1.00,
        0.94,
        0.86,
        0.82,
        0.86,
        0.92,
        0.74,
        0.48,
        0.26,
        0.16,
        0.12,
        0.10,
    ],
    "weekday_service",
)

WEEKEND_SERVICE = _validate_profile(
    [
        0.06,
        0.06,
        0.06,
        0.06,
        0.06,
        0.07,
        0.10,
        0.16,
        0.24,
        0.34,
        0.48,
        0.60,
        0.68,
        0.66,
        0.60,
        0.56,
        0.58,
        0.60,
        0.50,
        0.34,
        0.20,
        0.14,
        0.10,
        0.08,
    ],
    "weekend_service",
)

PV_DAY_SHAPE = _validate_profile(
    [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.02,
        0.08,
        0.18,
        0.35,
        0.54,
        0.72,
        0.86,
        0.96,
        1.00,
        0.90,
        0.72,
        0.48,
        0.22,
        0.05,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ],
    "pv_day_shape",
)

BUY_PRICE_DAY = _validate_profile(
    [
        0.46,
        0.44,
        0.43,
        0.43,
        0.45,
        0.50,
        0.58,
        0.69,
        0.80,
        0.86,
        0.88,
        0.83,
        0.76,
        0.72,
        0.74,
        0.82,
        0.92,
        1.02,
        1.08,
        1.04,
        0.92,
        0.76,
        0.62,
        0.52,
    ],
    "buy_price_day",
)

CARBON_INTENSITY_DAY = _validate_profile(
    [
        0.56,
        0.55,
        0.54,
        0.54,
        0.55,
        0.57,
        0.60,
        0.63,
        0.60,
        0.56,
        0.51,
        0.48,
        0.45,
        0.43,
        0.44,
        0.48,
        0.56,
        0.64,
        0.71,
        0.74,
        0.70,
        0.66,
        0.62,
        0.58,
    ],
    "carbon_intensity_day",
)

CARBON_PRICE_MULTIPLIER_DAY = _validate_profile(
    [
        0.96,
        0.95,
        0.95,
        0.95,
        0.96,
        0.99,
        1.02,
        1.05,
        1.08,
        1.09,
        1.08,
        1.04,
        1.00,
        0.98,
        0.99,
        1.02,
        1.07,
        1.12,
        1.16,
        1.18,
        1.14,
        1.08,
        1.02,
        0.99,
    ],
    "carbon_price_multiplier_day",
)

EV_WORKDAY_SHARE = _normalize(
    [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.01,
        0.03,
        0.06,
        0.08,
        0.10,
        0.11,
        0.11,
        0.09,
        0.08,
        0.08,
        0.09,
        0.08,
        0.04,
        0.02,
        0.01,
        0.01,
        0.00,
        0.00,
        0.00,
    ]
)

EV_WEEKEND_SHARE = _normalize(
    [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.11,
        0.11,
        0.10,
        0.10,
        0.09,
        0.07,
        0.05,
        0.03,
        0.02,
        0.01,
        0.00,
        0.00,
        0.00,
    ]
)


def _hourly_wave(hour: int, amplitude: float, phase_shift: float = 0.0) -> float:
    return 1.0 + amplitude * math.sin((hour / HOURS_PER_DAY) * 2.0 * math.pi + phase_shift)


def build_weekly_profiles(config: ScenarioConfig) -> dict[str, object]:
    if config.horizon_hours != HOURS_PER_WEEK:
        raise ValueError("The weekly profile library currently supports a fixed 7x24 horizon.")

    hvac_load, service_load = config.flexible_loads
    day_scalars = [0.97, 1.00, 1.03, 1.04, 1.01, 0.76, 0.70]
    hvac_temp_scalars = [0.98, 1.00, 1.04, 1.06, 1.05, 0.94, 0.90]
    service_scalars = [0.98, 1.00, 1.03, 1.04, 1.00, 0.62, 0.56]
    pv_weather_scalars = [0.78, 0.88, 0.94, 0.97, 0.91, 0.80, 0.74]
    price_day_scalars = [0.98, 1.00, 1.02, 1.04, 1.03, 0.88, 0.86]
    carbon_day_scalars = [0.99, 1.00, 1.01, 1.03, 1.02, 0.95, 0.94]
    temperature_by_day_c = [24.0, 25.0, 27.0, 28.0, 28.5, 25.5, 24.5]

    inflexible_profile: list[float] = []
    hvac_profile: list[float] = []
    service_profile: list[float] = []
    pv_profile: list[float] = []
    ev_profile: list[float] = []
    buy_price_profile: list[float] = []
    sell_price_profile: list[float] = []
    carbon_intensity_profile: list[float] = []
    carbon_price_profile: list[float] = []
    ambient_temperature_profile: list[float] = []
    day_index_profile: list[int] = []
    hour_of_day_profile: list[int] = []

    for day in range(DAYS_PER_WEEK):
        weekday = day < 5
        inflexible_shape = WEEKDAY_INFLEXIBLE if weekday else WEEKEND_INFLEXIBLE
        hvac_shape = WEEKDAY_HVAC if weekday else WEEKEND_HVAC
        service_shape = WEEKDAY_SERVICE if weekday else WEEKEND_SERVICE
        ev_share = EV_WORKDAY_SHARE if weekday else EV_WEEKEND_SHARE
        ev_daily_energy = config.ev_cluster.daily_energy_kwh * (1.0 if weekday else 0.62)
        day_temperature = temperature_by_day_c[day]
        temp_effect = 1.0 + max(day_temperature - 24.0, 0.0) * 0.018

        for hour in range(HOURS_PER_DAY):
            day_index_profile.append(day)
            hour_of_day_profile.append(hour)
            ambient_temperature = day_temperature + 3.5 * math.sin(((hour - 7) / HOURS_PER_DAY) * 2.0 * math.pi)
            ambient_temperature_profile.append(round(ambient_temperature, 3))

            load_wave = _hourly_wave(hour, amplitude=0.02, phase_shift=-0.4)
            price_wave = _hourly_wave(hour, amplitude=0.015, phase_shift=0.15)
            carbon_wave = _hourly_wave(hour, amplitude=0.012, phase_shift=0.35)

            inflexible_value = (
                config.inflexible_peak_kw
                * inflexible_shape[hour]
                * day_scalars[day]
                * load_wave
            )
            hvac_value = (
                hvac_load.baseline_peak_kw
                * hvac_shape[hour]
                * hvac_temp_scalars[day]
                * temp_effect
            )
            service_value = (
                service_load.baseline_peak_kw
                * service_shape[hour]
                * service_scalars[day]
                * load_wave
            )
            pv_value = (
                config.pv.rated_power_kw
                * PV_DAY_SHAPE[hour]
                * pv_weather_scalars[day]
                * max(0.92 + 0.03 * math.sin(hour / 3.0), 0.75)
            )
            ev_value = ev_daily_energy * ev_share[hour]
            buy_price = BUY_PRICE_DAY[hour] * price_day_scalars[day] * price_wave
            carbon_intensity = CARBON_INTENSITY_DAY[hour] * carbon_day_scalars[day] * carbon_wave
            carbon_price = 0.18 * CARBON_PRICE_MULTIPLIER_DAY[hour] * (0.98 + 0.015 * day)
            sell_price = max(0.28, buy_price * (0.66 if weekday else 0.64))

            inflexible_profile.append(round(inflexible_value, 3))
            hvac_profile.append(round(hvac_value, 3))
            service_profile.append(round(service_value, 3))
            pv_profile.append(round(pv_value, 3))
            ev_profile.append(round(ev_value, 3))
            buy_price_profile.append(round(buy_price, 4))
            sell_price_profile.append(round(sell_price, 4))
            carbon_intensity_profile.append(round(carbon_intensity, 4))
            carbon_price_profile.append(round(carbon_price, 4))

    return {
        "hours": list(range(HOURS_PER_WEEK)),
        "day_index": day_index_profile,
        "day_name": [DAY_NAMES[day] for day in day_index_profile],
        "hour_of_day": hour_of_day_profile,
        "ambient_temperature_c": ambient_temperature_profile,
        "inflexible_load_kw": inflexible_profile,
        "flexible_loads_kw": {
            hvac_load.name: hvac_profile,
            service_load.name: service_profile,
        },
        "pv_available_kw": pv_profile,
        "ev_energy_request_kwh": ev_profile,
        "buy_price_rmb_per_kwh": buy_price_profile,
        "sell_price_rmb_per_kwh": sell_price_profile,
        "carbon_price_rmb_per_kg": carbon_price_profile,
        "grid_carbon_intensity_kg_per_kwh": carbon_intensity_profile,
    }
