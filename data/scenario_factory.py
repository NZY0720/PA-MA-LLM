from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AssetSpec, EVClusterSpec, FlexibleLoadSpec, MarketSpec, ScenarioConfig
from .weekly_profiles import HOURS_PER_WEEK, build_weekly_profiles


def default_weekly_scenario_config(random_seed: int = 2026) -> ScenarioConfig:
    return ScenarioConfig(
        name="base_weekly_low_carbon_park",
        horizon_hours=HOURS_PER_WEEK,
        step_hours=1.0,
        random_seed=random_seed,
        operator_name="park_operator",
        tie_line_limit_kw=620.0,
        pv=AssetSpec(
            name="pv_unit",
            rated_power_kw=420.0,
            notes="Aggregated rooftop and parking-lot photovoltaic portfolio.",
        ),
        ess=AssetSpec(
            name="ess_unit",
            rated_power_kw=260.0,
            energy_capacity_kwh=720.0,
            notes="Lithium-ion battery with symmetric charge and discharge limits.",
        ),
        inflexible_peak_kw=460.0,
        flexible_loads=(
            FlexibleLoadSpec(
                name="hvac_load",
                baseline_peak_kw=140.0,
                adjustable_range_kw=140.0,
                daily_energy_kwh=1750.0,
                description="HVAC-dominant building load affected by weather and occupancy.",
            ),
            FlexibleLoadSpec(
                name="service_load",
                baseline_peak_kw=95.0,
                adjustable_range_kw=95.0,
                daily_energy_kwh=980.0,
                description="Shiftable service and public-area operation load.",
            ),
        ),
        ev_cluster=EVClusterSpec(
            slots=48,
            max_charging_power_kw=180.0,
            daily_energy_kwh=300.0,
            arrival_hour=7,
            departure_hour=20,
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
            "role": "Coordinate internal resources and external electricity-carbon interactions across a weekly horizon.",
        },
        "environment_agent": {
            "name": "market_environment",
            "role": "Broadcast weekly electricity price, carbon price, weather, and carbon-intensity signals.",
        },
        "resource_agents": [
            {"name": "pv_agent", "role": "Forecast renewable availability and curtailment opportunity."},
            {"name": "ess_agent", "role": "Manage arbitrage, reserve, and embodied-carbon shifting."},
            {"name": "ev_agent", "role": "Represent parking availability, charging urgency, and deferred demand."},
            {"name": "hvac_agent", "role": "Represent weather-sensitive thermal flexibility."},
            {"name": "service_load_agent", "role": "Represent shiftable service and occupancy-related demand."},
        ],
    }


def build_weekly_low_carbon_scenario(random_seed: int = 2026) -> dict[str, Any]:
    config = default_weekly_scenario_config(random_seed=random_seed)
    profiles = build_weekly_profiles(config)

    return {
        "scenario_name": config.name,
        "study_scope": {
            "time_scale": "week_ahead",
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
            "assumption": "Synthetic but week-realistic 7x24 profiles with weekday-weekend heterogeneity.",
            "reference_use": "Base weekly case for baseline comparison, ablation, and visualization.",
        },
    }


def export_weekly_low_carbon_scenario(output_path: str | Path, random_seed: int = 2026) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_weekly_low_carbon_scenario(random_seed=random_seed)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
