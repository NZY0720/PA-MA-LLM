from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io_utils import write_json

from .config import AssetSpec, EVClusterSpec, FlexibleLoadSpec, MarketSpec, ScenarioConfig
from .profiles import build_base_profiles


def default_scenario_config(random_seed: int = 2026) -> ScenarioConfig:
    return ScenarioConfig(
        name="base_day_ahead_low_carbon_park",
        horizon_hours=24,
        step_hours=1.0,
        random_seed=random_seed,
        operator_name="park_operator",
        tie_line_limit_kw=600.0,
        pv=AssetSpec(
            name="pv_unit",
            rated_power_kw=350.0,
            notes="Single aggregated distributed PV plant.",
        ),
        ess=AssetSpec(
            name="ess_unit",
            rated_power_kw=250.0,
            energy_capacity_kwh=500.0,
            notes="Battery ESS with symmetric charge/discharge limits.",
        ),
        inflexible_peak_kw=420.0,
        flexible_loads=(
            FlexibleLoadSpec(
                name="hvac_load",
                baseline_peak_kw=120.0,
                adjustable_range_kw=120.0,
                daily_energy_kwh=1700.0,
                description="HVAC-dominant building load with comfort-sensitive flexibility.",
            ),
            FlexibleLoadSpec(
                name="service_load",
                baseline_peak_kw=80.0,
                adjustable_range_kw=80.0,
                daily_energy_kwh=920.0,
                description="Shiftable service load for office and public-area operation.",
            ),
        ),
        ev_cluster=EVClusterSpec(
            slots=40,
            max_charging_power_kw=160.0,
            daily_energy_kwh=280.0,
            arrival_hour=7,
            departure_hour=18,
        ),
        market=MarketSpec(
            price_unit="RMB/kWh",
            carbon_price_unit="RMB/kg",
            carbon_intensity_unit="kg/kWh",
        ),
    )


def _build_agent_map(config: ScenarioConfig) -> dict[str, Any]:
    return {
        "operator_agent": {
            "name": config.operator_name,
            "role": "Coordinate internal resources and external electricity-carbon decisions.",
        },
        "environment_agent": {
            "name": "market_environment",
            "role": "Broadcast price, carbon, and forecast signals.",
        },
        "resource_agents": [
            {"name": "pv_agent", "role": "Provide renewable availability and curtailment preference."},
            {"name": "ess_agent", "role": "Manage arbitrage, reserve, and carbon-carrying states."},
            {"name": "ev_agent", "role": "Represent charging urgency and parking-window constraints."},
            {"name": "hvac_agent", "role": "Represent comfort-sensitive building flexibility."},
            {"name": "service_load_agent", "role": "Represent shiftable service-load preference."},
        ],
    }


def build_base_day_ahead_scenario(random_seed: int = 2026) -> dict[str, Any]:
    config = default_scenario_config(random_seed=random_seed)
    profiles = build_base_profiles(config)

    return {
        "scenario_name": config.name,
        "study_scope": {
            "time_scale": "day_ahead",
            "objective": [
                "internal_external_coordination",
                "physics_informed_feasibility",
                "dynamic_carbon_responsibility",
            ],
        },
        "config": config.to_dict(),
        "agents": _build_agent_map(config),
        "profiles": profiles,
        "notes": {
            "park_type": "office_commercial_low_carbon_park",
            "assumption": "Single tie-line, synthetic but realistic weekday profiles.",
            "reference_use": "Base case for baseline comparison, ablation, and visualization.",
        },
    }


def export_base_day_ahead_scenario(output_path: str | Path, random_seed: int = 2026) -> Path:
    target = Path(output_path)
    write_json(target, build_base_day_ahead_scenario(random_seed=random_seed))
    return target
