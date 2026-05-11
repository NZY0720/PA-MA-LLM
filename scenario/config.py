from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AssetSpec:
    name: str
    rated_power_kw: float
    energy_capacity_kwh: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class FlexibleLoadSpec:
    name: str
    baseline_peak_kw: float
    adjustable_range_kw: float
    daily_energy_kwh: float
    description: str


@dataclass(frozen=True)
class EVClusterSpec:
    slots: int
    max_charging_power_kw: float
    daily_energy_kwh: float
    arrival_hour: int
    departure_hour: int


@dataclass(frozen=True)
class MarketSpec:
    price_unit: str
    carbon_price_unit: str
    carbon_intensity_unit: str


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    horizon_hours: int
    step_hours: float
    random_seed: int
    operator_name: str
    tie_line_limit_kw: float
    pv: AssetSpec
    ess: AssetSpec
    inflexible_peak_kw: float
    flexible_loads: tuple[FlexibleLoadSpec, ...]
    ev_cluster: EVClusterSpec
    market: MarketSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
