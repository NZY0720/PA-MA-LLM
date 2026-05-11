from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.constants import HOURS_PER_DAY
from utils.math_utils import clamp, jains_index, series_payload, to_np

ESS_CH_EFF = 0.95
ESS_DIS_EFF = 0.95
ESS_MIN_RATIO = 0.10
ESS_INIT_RATIO = 0.50
ESS_DEG_COST = 0.015
LOAD_SHIFT_COST = 0.030
EV_SHORTAGE_PENALTY = 2.50
FLEX_SHORTAGE_PENALTY = 1.80
ESS_POWER_STEP = 20.0
ESS_ENERGY_STEP = 10.0

CASE2_PARK_IDS = ("Park_A", "Park_B", "Park_C", "Park_D", "Park_E")

_CARBON_QUOTA_FACTORS = {
    "Renewable-Rich Park": 1.05,
    "Load-Intensive Park": 0.88,
    "Storage-Dominant Park": 1.04,
    "Flexible-Demand Park": 0.98,
}


def carbon_quota_factor(park_type: str) -> float:
    return _CARBON_QUOTA_FACTORS.get(park_type, 0.96)


@dataclass
class BaselineRun:
    name: str
    run_id: str
    model_name: str
    total_operating_cost: float
    electricity_purchase_cost: float
    carbon_related_cost: float
    export_revenue: float
    service_penalty_cost: float
    battery_degradation_cost: float
    total_carbon_emission: float
    renewable_utilization: float
    high_carbon_grid_purchase: float
    carbon_responsibility_variance: float
    carbon_fairness_index: float
    constraint_violation_rate: float
    power_balance_error: float
    soc_violation_count: int
    run_to_run_reference_cost: float
    hvac_energy: float
    service_energy: float
    ev_energy: float
    details: dict[str, Any]


@dataclass
class Case2Run:
    name: str
    run_id: str
    model_name: str
    total_system_operating_cost: float
    total_system_carbon_emission: float
    total_grid_purchase: float
    interpark_trading_volume: float
    carbon_credit_trading_volume: float
    total_carbon_compliance_cost: float
    average_carbon_credit_price: float
    average_clearing_price: float
    grid_dependence_reduction: float
    negotiation_convergence_rate: float
    total_negotiation_rounds: float
    renewable_energy_sharing_rate: float
    high_carbon_exposure_reduction: float
    benefit_distribution_fairness: float
    carbon_responsibility_fairness: float
    winwin_ratio: float
    constraint_violation_rate: float
    power_balance_error: float
    park_costs: dict[str, float]
    trading_benefits: dict[str, float]
    park_emissions: dict[str, float]
    details: dict[str, Any]


def base_arrays(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    profiles = scenario["profiles"]
    return {
        "hours": to_np(profiles["hours"]),
        "day_index": to_np(profiles["day_index"]),
        "hour_of_day": to_np(profiles["hour_of_day"]),
        "temperature": to_np(profiles["ambient_temperature_c"]),
        "inflexible": to_np(profiles["inflexible_load_kw"]),
        "hvac_baseline": to_np(profiles["flexible_loads_kw"]["hvac_load"]),
        "service_baseline": to_np(profiles["flexible_loads_kw"]["service_load"]),
        "pv": to_np(profiles["pv_available_kw"]),
        "ev_request": to_np(profiles["ev_energy_request_kwh"]),
        "buy_price": to_np(profiles["buy_price_rmb_per_kwh"]),
        "sell_price": to_np(profiles["sell_price_rmb_per_kwh"]),
        "carbon_price": to_np(profiles["carbon_price_rmb_per_kg"]),
        "carbon_intensity": to_np(profiles["grid_carbon_intensity_kg_per_kwh"]),
    }


def aggregate_runs(name: str, runs: list[BaselineRun]) -> dict[str, Any]:
    metric_fields = [
        "total_operating_cost",
        "electricity_purchase_cost",
        "carbon_related_cost",
        "total_carbon_emission",
        "renewable_utilization",
        "high_carbon_grid_purchase",
        "carbon_responsibility_variance",
        "carbon_fairness_index",
        "constraint_violation_rate",
        "power_balance_error",
    ]
    summary: dict[str, Any] = {"baseline": name, "runs": [run.__dict__ for run in runs], "metrics": {}}
    for field in metric_fields:
        values = [getattr(run, field) for run in runs]
        summary["metrics"][field] = {
            "mean": round(float(statistics.mean(values)), 6),
            "std": round(float(statistics.pstdev(values)), 6) if len(values) > 1 else 0.0,
        }
    soc_values = [run.soc_violation_count for run in runs]
    summary["metrics"]["soc_violation_count"] = {
        "mean": round(float(statistics.mean(soc_values)), 6),
        "std": round(float(statistics.pstdev(soc_values)), 6) if len(soc_values) > 1 else 0.0,
    }
    return summary


def aggregate_case2_runs(name: str, runs: list[Case2Run]) -> dict[str, Any]:
    metric_fields = [
        "total_system_operating_cost",
        "total_system_carbon_emission",
        "total_grid_purchase",
        "interpark_trading_volume",
        "carbon_credit_trading_volume",
        "total_carbon_compliance_cost",
        "average_carbon_credit_price",
        "average_clearing_price",
        "grid_dependence_reduction",
        "negotiation_convergence_rate",
        "total_negotiation_rounds",
        "renewable_energy_sharing_rate",
        "high_carbon_exposure_reduction",
        "benefit_distribution_fairness",
        "carbon_responsibility_fairness",
        "winwin_ratio",
        "constraint_violation_rate",
        "power_balance_error",
    ]
    summary: dict[str, Any] = {"baseline": name, "runs": [run.__dict__ for run in runs], "metrics": {}}
    for field in metric_fields:
        values = [float(getattr(run, field)) for run in runs]
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            finite_values = [0.0]
        summary["metrics"][field] = {
            "mean": round(float(statistics.mean(finite_values)), 6),
            "std": round(float(statistics.pstdev(finite_values)), 6) if len(finite_values) > 1 else 0.0,
        }
    return summary


def select_reference_run(runs: list[BaselineRun]) -> BaselineRun:
    return sorted(runs, key=lambda run: (run.constraint_violation_rate, run.total_operating_cost))[0]


def select_reference_case2_run(runs: list[Case2Run]) -> Case2Run:
    return sorted(runs, key=lambda run: (run.constraint_violation_rate, run.total_system_operating_cost))[0]


def allocate_energy(required: float, lower: np.ndarray, upper: np.ndarray, score: np.ndarray) -> np.ndarray:
    schedule = lower.astype(float).copy()
    remaining = required - float(schedule.sum())
    if remaining < -1e-6:
        raise ValueError("Lower bounds already exceed the required energy.")

    order = np.argsort(score)
    for idx in order:
        if remaining <= 1e-9:
            break
        room = float(upper[idx] - schedule[idx])
        if room <= 1e-9:
            continue
        add = min(room, remaining)
        schedule[idx] += add
        remaining -= add

    if remaining > 1e-4:
        raise ValueError("Unable to allocate the required energy within the admissible bounds.")
    return schedule


def _quantize_energy(value: float, energy_min: float, energy_max: float) -> float:
    value = clamp(value, energy_min, energy_max)
    return round(value / ESS_ENERGY_STEP) * ESS_ENERGY_STEP


def optimize_ess_schedule(
    net_load_without_ess: np.ndarray,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    carbon_price: np.ndarray,
    carbon_intensity: np.ndarray,
    tie_line_limit: float,
    power_limit: float,
    energy_capacity: float,
    desired_ess_power: np.ndarray | None = None,
    carbon_multiplier: float = 1.0,
    projection_weight: float = 0.0,
) -> np.ndarray:
    horizon = len(net_load_without_ess)
    energy_min = ESS_MIN_RATIO * energy_capacity
    energy_init = ESS_INIT_RATIO * energy_capacity
    energy_target = energy_init

    action_values = np.arange(-power_limit, power_limit + ESS_POWER_STEP, ESS_POWER_STEP, dtype=float)
    state_values = np.arange(0.0, energy_capacity + ESS_ENERGY_STEP, ESS_ENERGY_STEP, dtype=float)
    state_values = state_values[state_values >= energy_min - 1e-9]
    def nearest_state_idx(value: float) -> int:
        clipped = clamp(value, float(state_values[0]), float(state_values[-1]))
        return int(np.argmin(np.abs(state_values - clipped)))

    target_state = nearest_state_idx(energy_target)
    init_state = nearest_state_idx(energy_init)

    large_cost = 1e18
    dp = np.full((horizon + 1, len(state_values)), large_cost, dtype=float)
    prev_state = np.full((horizon + 1, len(state_values)), -1, dtype=int)
    prev_action = np.full((horizon + 1, len(state_values)), np.nan, dtype=float)
    dp[0, init_state] = 0.0

    desired = desired_ess_power if desired_ess_power is not None else np.zeros(horizon, dtype=float)
    for t in range(horizon):
        for s_idx, energy in enumerate(state_values):
            if dp[t, s_idx] >= large_cost / 10:
                continue
            for action in action_values:
                if action >= 0.0:
                    next_energy = energy - action / ESS_DIS_EFF
                else:
                    next_energy = energy + (-action) * ESS_CH_EFF

                if next_energy < energy_min - 1e-9 or next_energy > energy_capacity + 1e-9:
                    continue

                grid_exchange = net_load_without_ess[t] - action
                if abs(grid_exchange) > tie_line_limit + 1e-9:
                    continue

                buy_kw = max(grid_exchange, 0.0)
                sell_kw = max(-grid_exchange, 0.0)
                op_cost = (
                    buy_kw * buy_price[t]
                    - sell_kw * sell_price[t]
                    + carbon_multiplier * buy_kw * carbon_intensity[t] * carbon_price[t]
                    + ESS_DEG_COST * abs(action)
                )
                projection_cost = projection_weight * ((action - desired[t]) / max(power_limit, 1.0)) ** 2
                next_idx = nearest_state_idx(next_energy)
                total_cost = dp[t, s_idx] + op_cost + projection_cost
                if total_cost + 1e-9 < dp[t + 1, next_idx]:
                    dp[t + 1, next_idx] = total_cost
                    prev_state[t + 1, next_idx] = s_idx
                    prev_action[t + 1, next_idx] = action

    best_terminal = None
    best_cost = large_cost
    for s_idx, energy in enumerate(state_values):
        terminal_penalty = 0.10 * abs(energy - state_values[target_state])
        total_cost = dp[horizon, s_idx] + terminal_penalty
        if total_cost < best_cost:
            best_cost = total_cost
            best_terminal = s_idx

    if best_terminal is None:
        raise RuntimeError("No feasible ESS trajectory was found.")

    schedule = np.zeros(horizon, dtype=float)
    cursor = best_terminal
    for t in range(horizon, 0, -1):
        schedule[t - 1] = prev_action[t, cursor]
        cursor = prev_state[t, cursor]
        if cursor < 0 and t > 1:
            raise RuntimeError("Failed to reconstruct the ESS schedule path.")
    return schedule


def simulate_internal_plan(
    name: str,
    run_id: str,
    scenario: dict[str, Any],
    plan: dict[str, np.ndarray],
    llm_intent: dict[str, Any] | None,
    physics_projection: bool,
) -> BaselineRun:
    arrays = base_arrays(scenario)
    config = scenario["config"]
    horizon = len(arrays["hours"])

    tie_line_limit = float(config["tie_line_limit_kw"])
    ess_power_limit = float(config["ess"]["rated_power_kw"])
    ess_energy_capacity = float(config["ess"]["energy_capacity_kwh"])
    ess_energy_min = ESS_MIN_RATIO * ess_energy_capacity
    ess_energy = ESS_INIT_RATIO * ess_energy_capacity
    ev_arrival = int(config["ev_cluster"]["arrival_hour"])
    ev_departure = int(config["ev_cluster"]["departure_hour"])
    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    ev_power_max = float(config["ev_cluster"]["max_charging_power_kw"])
    hvac_target = float(config["flexible_loads"][0]["daily_energy_kwh"]) * (horizon / HOURS_PER_DAY)
    service_target = float(config["flexible_loads"][1]["daily_energy_kwh"]) * (horizon / HOURS_PER_DAY)
    ev_target = float(np.sum(arrays["ev_request"]))

    hvac = np.asarray(plan["hvac"], dtype=float).copy()
    service = np.asarray(plan["service"], dtype=float).copy()
    ev = np.asarray(plan["ev"], dtype=float).copy()
    ess = np.asarray(plan["ess"], dtype=float).copy()

    violations = 0
    total_checks = horizon * 4 + 3
    soc_violation_count = 0
    in_window = (arrays["hour_of_day"] >= ev_arrival) & (arrays["hour_of_day"] <= ev_departure)

    if not physics_projection:
        violations += int(np.count_nonzero(hvac < 0.0) + np.count_nonzero(hvac > hvac_peak))
        violations += int(np.count_nonzero(service < 0.0) + np.count_nonzero(service > service_peak))
        violations += int(np.count_nonzero(ev[~in_window] > 1e-6))
        hvac = np.clip(hvac, 0.0, hvac_peak)
        service = np.clip(service, 0.0, service_peak)
        ev = np.clip(ev, 0.0, ev_power_max)
    else:
        hvac = np.clip(hvac, 0.0, hvac_peak)
        service = np.clip(service, 0.0, service_peak)
        ev = np.clip(ev, 0.0, ev_power_max)
        ev[~in_window] = 0.0

    load_total = arrays["inflexible"] + hvac + service + ev
    buy = np.zeros(horizon, dtype=float)
    sell = np.zeros(horizon, dtype=float)
    pv_direct = np.zeros(horizon, dtype=float)
    pv_to_charge = np.zeros(horizon, dtype=float)
    pv_export = np.zeros(horizon, dtype=float)
    pv_curtailment = np.zeros(horizon, dtype=float)
    ess_dis_to_load = np.zeros(horizon, dtype=float)
    grid_to_load = np.zeros(horizon, dtype=float)
    grid_to_charge = np.zeros(horizon, dtype=float)
    soc_series = np.zeros(horizon, dtype=float)
    ess_carbon_avg = np.zeros(horizon, dtype=float)
    candidate_balance_error = np.zeros(horizon, dtype=float)

    entity_names = ["inflexible_load", "hvac_load", "service_load", "ev_cluster"]
    entity_loads = {
        "inflexible_load": arrays["inflexible"].copy(),
        "hvac_load": hvac.copy(),
        "service_load": service.copy(),
        "ev_cluster": ev.copy(),
    }
    hourly_carbon: dict[str, np.ndarray] = {entity: np.zeros(horizon, dtype=float) for entity in entity_names}
    ess_carbon_content = 0.0

    for t in range(horizon):
        candidate_ess = float(ess[t])
        if abs(candidate_ess) > ess_power_limit + 1e-9 and not physics_projection:
            violations += 1

        candidate_grid = float(load_total[t] - arrays["pv"][t] - candidate_ess)
        candidate_balance_error[t] = abs(candidate_grid) if not physics_projection else 0.0
        if abs(candidate_grid) > tie_line_limit + 1e-9 and not physics_projection:
            violations += 1

        actual_ess = clamp(candidate_ess, -ess_power_limit, ess_power_limit) if physics_projection else candidate_ess
        if actual_ess >= 0.0:
            max_discharge = max(0.0, (ess_energy - ess_energy_min) * ESS_DIS_EFF)
            if actual_ess > max_discharge + 1e-9:
                if physics_projection:
                    actual_ess = max_discharge
                else:
                    soc_violation_count += 1
                    violations += 1
            ess_energy_after = ess_energy - actual_ess / ESS_DIS_EFF
        else:
            max_charge = max(0.0, (ess_energy_capacity - ess_energy) / ESS_CH_EFF)
            if -actual_ess > max_charge + 1e-9:
                if physics_projection:
                    actual_ess = -max_charge
                else:
                    soc_violation_count += 1
                    violations += 1
            ess_energy_after = ess_energy + (-actual_ess) * ESS_CH_EFF

        if ess_energy_after < ess_energy_min - 1e-9 or ess_energy_after > ess_energy_capacity + 1e-9:
            if physics_projection:
                ess_energy_after = clamp(ess_energy_after, ess_energy_min, ess_energy_capacity)
            else:
                soc_violation_count += 1
                violations += 1

        raw_grid = load_total[t] - arrays["pv"][t] - actual_ess
        buy[t] = max(raw_grid, 0.0)
        sell[t] = max(-raw_grid, 0.0)
        if buy[t] > tie_line_limit + 1e-9 or sell[t] > tie_line_limit + 1e-9:
            violations += int(not physics_projection)
        if physics_projection:
            buy[t] = min(buy[t], tie_line_limit)
            sell[t] = min(sell[t], tie_line_limit)

        if actual_ess >= 0.0:
            pv_direct[t] = min(arrays["pv"][t], load_total[t])
            remaining_load = max(load_total[t] - pv_direct[t], 0.0)
            ess_dis_to_load[t] = min(actual_ess, remaining_load)
            remaining_load = max(remaining_load - ess_dis_to_load[t], 0.0)
            grid_to_load[t] = remaining_load
            pv_surplus = max(arrays["pv"][t] - pv_direct[t], 0.0)
            pv_export[t] = min(sell[t], pv_surplus)
            pv_curtailment[t] = max(arrays["pv"][t] - pv_direct[t] - pv_export[t], 0.0)
            ess_avg_prev = ess_carbon_content / max(ess_energy, 1e-6)
            ess_carbon_content = max(0.0, ess_carbon_content - ess_avg_prev * actual_ess)
        else:
            charge_power = -actual_ess
            pv_direct[t] = min(arrays["pv"][t], load_total[t])
            remaining_load = max(load_total[t] - pv_direct[t], 0.0)
            grid_to_load[t] = remaining_load
            pv_surplus = max(arrays["pv"][t] - pv_direct[t], 0.0)
            pv_to_charge[t] = min(pv_surplus, charge_power)
            grid_to_charge[t] = max(charge_power - pv_to_charge[t], 0.0)
            pv_export[t] = min(max(sell[t], 0.0), max(pv_surplus - pv_to_charge[t], 0.0))
            pv_curtailment[t] = max(arrays["pv"][t] - pv_direct[t] - pv_to_charge[t] - pv_export[t], 0.0)
            charge_intensity = arrays["carbon_intensity"][t] * (grid_to_charge[t] / max(charge_power, 1e-6))
            ess_carbon_content += charge_intensity * charge_power * ESS_CH_EFF

        ess_energy = clamp(ess_energy_after, ess_energy_min, ess_energy_capacity) if physics_projection else ess_energy_after
        ess_carbon_avg[t] = ess_carbon_content / max(ess_energy, 1e-6) if ess_energy > 1e-6 else 0.0
        soc_series[t] = ess_energy

        total_served_load = max(load_total[t], 1e-6)
        direct_carbon = grid_to_load[t] * arrays["carbon_intensity"][t] + ess_dis_to_load[t] * (
            ess_carbon_avg[t - 1] if t > 0 else 0.0
        )
        for entity in entity_names:
            share = entity_loads[entity][t] / total_served_load
            hourly_carbon[entity][t] = share * direct_carbon

    electricity_purchase_cost = float(np.sum(buy * arrays["buy_price"]))
    export_revenue = float(np.sum(sell * arrays["sell_price"]))
    carbon_cost = float(np.sum(buy * arrays["carbon_intensity"] * arrays["carbon_price"]))
    battery_degradation_cost = float(np.sum(np.abs(ess)) * ESS_DEG_COST)
    service_penalty_cost = (
        LOAD_SHIFT_COST * float(np.sum(np.abs(hvac - arrays["hvac_baseline"])))
        + LOAD_SHIFT_COST * float(np.sum(np.abs(service - arrays["service_baseline"])))
        + FLEX_SHORTAGE_PENALTY * abs(float(hvac.sum()) - hvac_target)
        + FLEX_SHORTAGE_PENALTY * abs(float(service.sum()) - service_target)
        + EV_SHORTAGE_PENALTY * abs(float(ev.sum()) - ev_target)
    )
    total_cost = electricity_purchase_cost - export_revenue + carbon_cost + battery_degradation_cost + service_penalty_cost
    total_emission = float(np.sum(buy * arrays["carbon_intensity"]))
    pv_used = float(np.sum(pv_direct + pv_to_charge))
    renewable_utilization = pv_used / max(float(load_total.sum()), 1e-6)
    carbon_threshold = float(np.quantile(arrays["carbon_intensity"], 0.75))
    high_carbon_purchase = float(np.sum(buy[arrays["carbon_intensity"] >= carbon_threshold]))

    entity_total_energy = np.array(
        [
            float(entity_loads["inflexible_load"].sum()),
            float(entity_loads["hvac_load"].sum()),
            float(entity_loads["service_load"].sum()),
            float(entity_loads["ev_cluster"].sum()),
        ],
        dtype=float,
    )
    entity_total_carbon = np.array(
        [
            float(hourly_carbon["inflexible_load"].sum()),
            float(hourly_carbon["hvac_load"].sum()),
            float(hourly_carbon["service_load"].sum()),
            float(hourly_carbon["ev_cluster"].sum()),
        ],
        dtype=float,
    )
    entity_intensity = entity_total_carbon / np.maximum(entity_total_energy, 1e-6)
    energy_share = entity_total_energy / max(float(entity_total_energy.sum()), 1e-6)
    carbon_share = entity_total_carbon / max(float(entity_total_carbon.sum()), 1e-6)
    burden_ratio = carbon_share / np.maximum(energy_share, 1e-6)
    carbon_variance = float(np.var(burden_ratio))
    carbon_fairness = jains_index(burden_ratio)

    if abs(float(hvac.sum()) - hvac_target) > 1e-3:
        violations += 1
    if abs(float(service.sum()) - service_target) > 1e-3:
        violations += 1
    if abs(float(ev.sum()) - ev_target) > 1e-3:
        violations += 1

    result_details = {
        "hourly": {
            "buy_kw": series_payload(buy),
            "sell_kw": series_payload(sell),
            "pv_direct_kw": series_payload(pv_direct),
            "pv_to_charge_kw": series_payload(pv_to_charge),
            "pv_export_kw": series_payload(pv_export),
            "pv_curtailment_kw": series_payload(pv_curtailment),
            "ess_kw": series_payload(ess),
            "soc_kwh": series_payload(soc_series),
            "hvac_kw": series_payload(hvac),
            "service_kw": series_payload(service),
            "ev_kw": series_payload(ev),
            "total_load_kw": series_payload(load_total),
            "candidate_balance_error_kw": series_payload(candidate_balance_error),
        },
        "carbon_responsibility": {
            entity: series_payload(hourly_carbon[entity]) for entity in entity_names
        },
        "entity_carbon_intensity": {
            entity: round(float(value), 4)
            for entity, value in zip(entity_names, entity_intensity.tolist())
        },
        "entity_carbon_burden_ratio": {
            entity: round(float(value), 4)
            for entity, value in zip(entity_names, burden_ratio.tolist())
        },
        "entity_total_carbon": {
            entity: round(float(value), 4)
            for entity, value in zip(entity_names, entity_total_carbon.tolist())
        },
        "llm_summary": llm_intent.get("summary", []) if llm_intent is not None else [],
        "agent_rationales": llm_intent.get("agent_rationales", {}) if llm_intent is not None else {},
    }

    return BaselineRun(
        name=name,
        run_id=run_id,
        model_name=llm_intent.get("model_name", "deterministic") if llm_intent is not None else "deterministic",
        total_operating_cost=round(total_cost, 4),
        electricity_purchase_cost=round(electricity_purchase_cost, 4),
        carbon_related_cost=round(carbon_cost, 4),
        export_revenue=round(export_revenue, 4),
        service_penalty_cost=round(service_penalty_cost, 4),
        battery_degradation_cost=round(battery_degradation_cost, 4),
        total_carbon_emission=round(total_emission, 4),
        renewable_utilization=round(renewable_utilization, 4),
        high_carbon_grid_purchase=round(high_carbon_purchase, 4),
        carbon_responsibility_variance=round(carbon_variance, 6),
        carbon_fairness_index=round(carbon_fairness, 6),
        constraint_violation_rate=round(float(violations / max(total_checks, 1)), 6),
        power_balance_error=round(float(candidate_balance_error.mean()), 4),
        soc_violation_count=int(soc_violation_count),
        run_to_run_reference_cost=round(total_cost, 4),
        hvac_energy=round(float(hvac.sum()), 4),
        service_energy=round(float(service.sum()), 4),
        ev_energy=round(float(ev.sum()), 4),
        details=result_details,
    )
