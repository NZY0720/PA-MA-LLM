from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from data.scenario_factory import export_weekly_low_carbon_scenario
from methods.common import (
    BaselineRun,
    aggregate_runs,
    allocate_energy,
    base_arrays,
    optimize_ess_schedule,
    select_reference_run,
    simulate_internal_plan,
)
from methods.llm_agents import Case1MultiAgentOrchestrator
from utils.constants import CASE1_OUTPUT_DIR, DEFAULT_MODEL, HOURS_PER_DAY, KEY_PATH, SCENARIO_PATH
from utils.io_utils import load_json, read_api_key, write_json, write_markdown
from utils.math_utils import moving_average, series_payload
from visualization.case1 import plot_case1_dispatch


FIGURE_DIR = CASE1_OUTPUT_DIR / "figures"
LLM_DIR = CASE1_OUTPUT_DIR / "llm_traces"


def _load_or_build_scenario() -> dict[str, Any]:
    if not SCENARIO_PATH.exists():
        export_weekly_low_carbon_scenario(SCENARIO_PATH)
    return load_json(SCENARIO_PATH)


def _daily_scaled_target(config_value: float, horizon: int) -> float:
    return float(config_value) * (horizon / HOURS_PER_DAY)


def _build_rule_based_plan(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    config = scenario["config"]
    arrays = base_arrays(scenario)
    horizon = len(arrays["hours"])

    pv = arrays["pv"]
    price = arrays["buy_price"]
    carbon = arrays["carbon_intensity"]
    hvac_base = arrays["hvac_baseline"]
    service_base = arrays["service_baseline"]

    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    hvac_target = _daily_scaled_target(config["flexible_loads"][0]["daily_energy_kwh"], horizon)
    service_target = _daily_scaled_target(config["flexible_loads"][1]["daily_energy_kwh"], horizon)
    ev_target = float(np.sum(arrays["ev_request"]))
    ev_power_max = float(config["ev_cluster"]["max_charging_power_kw"])
    ev_arrival = int(config["ev_cluster"]["arrival_hour"])
    ev_departure = int(config["ev_cluster"]["departure_hour"])

    midday_score = -(0.60 * (pv / max(float(pv.max()), 1.0)) + 0.25 * (1.0 - price / max(float(price.max()), 1.0)))
    evening_penalty = np.where((arrays["hour_of_day"] >= 17) & (arrays["hour_of_day"] <= 21), 0.22, 0.0)
    effective_score = midday_score + evening_penalty + 0.18 * (carbon / max(float(carbon.max()), 1.0))

    hvac_lower = np.maximum(0.60 * hvac_base, hvac_base - 0.20 * hvac_peak)
    hvac_upper = np.minimum(hvac_peak, hvac_base + 0.20 * hvac_peak)
    service_lower = np.maximum(0.35 * service_base, service_base - 0.30 * service_peak)
    service_upper = np.minimum(service_peak, service_base + 0.35 * service_peak)

    hvac = allocate_energy(hvac_target, hvac_lower, hvac_upper, effective_score)
    service = allocate_energy(service_target, service_lower, service_upper, effective_score - 0.10 * pv / max(float(pv.max()), 1.0))

    ev_lower = np.zeros(horizon, dtype=float)
    ev_upper = np.zeros(horizon, dtype=float)
    ev_upper[(arrays["hour_of_day"] >= ev_arrival) & (arrays["hour_of_day"] <= ev_departure)] = ev_power_max
    ev_score = price + carbon * 0.10 - 0.40 * (pv / max(float(pv.max()), 1.0))
    ev = allocate_energy(ev_target, ev_lower, ev_upper, ev_score)

    load_without_ess = arrays["inflexible"] + hvac + service + ev
    tie_line_limit = float(config["tie_line_limit_kw"])
    ess_power = float(config["ess"]["rated_power_kw"])
    ess_energy = float(config["ess"]["energy_capacity_kwh"])
    desired_ess = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        if pv[t] > load_without_ess[t]:
            desired_ess[t] = -min(ess_power * 0.75, pv[t] - load_without_ess[t])
        elif price[t] <= np.quantile(price, 0.25) and carbon[t] <= np.quantile(carbon, 0.35):
            desired_ess[t] = -0.30 * ess_power
        elif price[t] >= np.quantile(price, 0.80) or carbon[t] >= np.quantile(carbon, 0.80):
            desired_ess[t] = 0.55 * ess_power

    ess = optimize_ess_schedule(
        net_load_without_ess=load_without_ess - pv,
        buy_price=price,
        sell_price=arrays["sell_price"],
        carbon_price=arrays["carbon_price"],
        carbon_intensity=carbon,
        tie_line_limit=tie_line_limit,
        power_limit=ess_power,
        energy_capacity=ess_energy,
        desired_ess_power=desired_ess,
        carbon_multiplier=1.0,
        projection_weight=18.0,
    )
    return {"hvac": hvac, "service": service, "ev": ev, "ess": ess}


def _build_centralized_plan(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    config = scenario["config"]
    arrays = base_arrays(scenario)
    horizon = len(arrays["hours"])

    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    hvac_target = _daily_scaled_target(config["flexible_loads"][0]["daily_energy_kwh"], horizon)
    service_target = _daily_scaled_target(config["flexible_loads"][1]["daily_energy_kwh"], horizon)
    ev_target = float(np.sum(arrays["ev_request"]))
    ev_power_max = float(config["ev_cluster"]["max_charging_power_kw"])
    ev_arrival = int(config["ev_cluster"]["arrival_hour"])
    ev_departure = int(config["ev_cluster"]["departure_hour"])

    effective_cost = arrays["buy_price"] + arrays["carbon_price"] * arrays["carbon_intensity"]
    pv_bonus = 0.22 * (arrays["pv"] / max(float(arrays["pv"].max()), 1.0))

    hvac_lower = np.maximum(0.52 * arrays["hvac_baseline"], arrays["hvac_baseline"] - 0.22 * hvac_peak)
    hvac_upper = np.minimum(hvac_peak, arrays["hvac_baseline"] + 0.22 * hvac_peak)
    service_lower = np.maximum(0.28 * arrays["service_baseline"], arrays["service_baseline"] - 0.36 * service_peak)
    service_upper = np.minimum(service_peak, arrays["service_baseline"] + 0.36 * service_peak)
    hvac = allocate_energy(hvac_target, hvac_lower, hvac_upper, effective_cost - pv_bonus)
    service = allocate_energy(service_target, service_lower, service_upper, effective_cost - 1.35 * pv_bonus)

    ev_lower = np.zeros(horizon, dtype=float)
    ev_upper = np.zeros(horizon, dtype=float)
    ev_upper[(arrays["hour_of_day"] >= ev_arrival) & (arrays["hour_of_day"] <= ev_departure)] = ev_power_max
    ev = allocate_energy(ev_target, ev_lower, ev_upper, effective_cost - 1.65 * pv_bonus)

    ess = optimize_ess_schedule(
        net_load_without_ess=arrays["inflexible"] + hvac + service + ev - arrays["pv"],
        buy_price=arrays["buy_price"],
        sell_price=arrays["sell_price"],
        carbon_price=arrays["carbon_price"],
        carbon_intensity=arrays["carbon_intensity"],
        tie_line_limit=float(config["tie_line_limit_kw"]),
        power_limit=float(config["ess"]["rated_power_kw"]),
        energy_capacity=float(config["ess"]["energy_capacity_kwh"]),
        desired_ess_power=np.zeros(horizon, dtype=float),
        carbon_multiplier=1.08,
        projection_weight=0.0,
    )
    return {"hvac": hvac, "service": service, "ev": ev, "ess": ess}


def _build_raw_llm_plan(scenario: dict[str, Any], llm_intent: dict[str, Any]) -> dict[str, np.ndarray]:
    config = scenario["config"]
    arrays = base_arrays(scenario)
    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    ev_power = float(config["ev_cluster"]["max_charging_power_kw"])
    ess_power = float(config["ess"]["rated_power_kw"])
    return {
        "hvac": arrays["hvac_baseline"] + llm_intent["hvac_signal"] * 0.34 * hvac_peak,
        "service": arrays["service_baseline"] + llm_intent["service_signal"] * 0.42 * service_peak,
        "ev": llm_intent["ev_signal"] * ev_power,
        "ess": -llm_intent["ess_signal"] * ess_power * 0.95,
    }


def _build_pi_plan(scenario: dict[str, Any], llm_intent: dict[str, Any]) -> dict[str, np.ndarray]:
    config = scenario["config"]
    arrays = base_arrays(scenario)
    horizon = len(arrays["hours"])

    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    hvac_target = _daily_scaled_target(config["flexible_loads"][0]["daily_energy_kwh"], horizon)
    service_target = _daily_scaled_target(config["flexible_loads"][1]["daily_energy_kwh"], horizon)
    ev_target = float(np.sum(arrays["ev_request"]))
    ev_power = float(config["ev_cluster"]["max_charging_power_kw"])
    ev_arrival = int(config["ev_cluster"]["arrival_hour"])
    ev_departure = int(config["ev_cluster"]["departure_hour"])

    econ_focus = llm_intent["operator_econ_focus"]
    carbon_focus = llm_intent["operator_carbon_focus"]
    price_like = arrays["buy_price"] * (0.82 + 0.62 * econ_focus)
    carbon_like = arrays["carbon_price"] * arrays["carbon_intensity"] * (0.90 + 1.45 * carbon_focus)
    effective_score = price_like + carbon_like - 0.28 * arrays["pv"] / max(float(arrays["pv"].max()), 1.0)

    hvac_lower = np.maximum(0.58 * arrays["hvac_baseline"], arrays["hvac_baseline"] - 0.22 * hvac_peak)
    hvac_upper = np.minimum(hvac_peak, arrays["hvac_baseline"] + 0.22 * hvac_peak)
    service_lower = np.maximum(0.32 * arrays["service_baseline"], arrays["service_baseline"] - 0.36 * service_peak)
    service_upper = np.minimum(service_peak, arrays["service_baseline"] + 0.40 * service_peak)
    hvac = allocate_energy(hvac_target, hvac_lower, hvac_upper, effective_score - 0.20 * llm_intent["hvac_signal"])
    service = allocate_energy(service_target, service_lower, service_upper, effective_score - 0.24 * llm_intent["service_signal"])

    ev_lower = np.zeros(horizon, dtype=float)
    ev_upper = np.zeros(horizon, dtype=float)
    ev_upper[(arrays["hour_of_day"] >= ev_arrival) & (arrays["hour_of_day"] <= ev_departure)] = ev_power
    ev = allocate_energy(ev_target, ev_lower, ev_upper, effective_score - 0.36 * llm_intent["ev_signal"])

    carbon_multiplier = float(1.00 + 0.85 * carbon_focus.mean() - 0.18 * econ_focus.mean())
    ess = optimize_ess_schedule(
        net_load_without_ess=arrays["inflexible"] + hvac + service + ev - arrays["pv"],
        buy_price=arrays["buy_price"],
        sell_price=arrays["sell_price"],
        carbon_price=arrays["carbon_price"],
        carbon_intensity=arrays["carbon_intensity"],
        tie_line_limit=float(config["tie_line_limit_kw"]),
        power_limit=float(config["ess"]["rated_power_kw"]),
        energy_capacity=float(config["ess"]["energy_capacity_kwh"]),
        desired_ess_power=-llm_intent["ess_signal"] * float(config["ess"]["rated_power_kw"]) * 0.72,
        carbon_multiplier=carbon_multiplier,
        projection_weight=28.0,
    )
    return {"hvac": hvac, "service": service, "ev": ev, "ess": ess}


def _build_single_llm_plan(scenario: dict[str, Any], llm_intent: dict[str, Any]) -> dict[str, np.ndarray]:
    config = scenario["config"]
    arrays = base_arrays(scenario)
    horizon = len(arrays["hours"])

    hvac_peak = float(config["flexible_loads"][0]["baseline_peak_kw"])
    service_peak = float(config["flexible_loads"][1]["baseline_peak_kw"])
    hvac_target = _daily_scaled_target(config["flexible_loads"][0]["daily_energy_kwh"], horizon)
    service_target = _daily_scaled_target(config["flexible_loads"][1]["daily_energy_kwh"], horizon)
    ev_target = float(np.sum(arrays["ev_request"]))
    ev_power = float(config["ev_cluster"]["max_charging_power_kw"])
    ev_arrival = int(config["ev_cluster"]["arrival_hour"])
    ev_departure = int(config["ev_cluster"]["departure_hour"])

    base_unified_score = arrays["buy_price"] * (0.90 + 0.40 * llm_intent["operator_econ_focus"]) + arrays["carbon_price"] * arrays["carbon_intensity"] * (0.90 + 1.05 * llm_intent["operator_carbon_focus"])
    unified_score = moving_average(base_unified_score, window=12)

    hvac_lower = np.maximum(0.58 * arrays["hvac_baseline"], arrays["hvac_baseline"] - 0.20 * hvac_peak)
    hvac_upper = np.minimum(hvac_peak, arrays["hvac_baseline"] + 0.20 * hvac_peak)
    service_lower = np.maximum(0.30 * arrays["service_baseline"], arrays["service_baseline"] - 0.35 * service_peak)
    service_upper = np.minimum(service_peak, arrays["service_baseline"] + 0.38 * service_peak)
    hvac = allocate_energy(hvac_target, hvac_lower, hvac_upper, unified_score)
    service = allocate_energy(service_target, service_lower, service_upper, unified_score)

    ev_lower = np.zeros(horizon, dtype=float)
    ev_upper = np.zeros(horizon, dtype=float)
    ev_upper[(arrays["hour_of_day"] >= ev_arrival) & (arrays["hour_of_day"] <= ev_departure)] = ev_power
    ev = allocate_energy(ev_target, ev_lower, ev_upper, unified_score)

    normalized_score = unified_score - float(unified_score.mean())
    score_scale = max(float(np.std(normalized_score)), 1e-6)
    shared_ess_signal = np.clip(normalized_score / score_scale, -1.0, 1.0)
    ess = optimize_ess_schedule(
        net_load_without_ess=arrays["inflexible"] + hvac + service + ev - arrays["pv"],
        buy_price=arrays["buy_price"],
        sell_price=arrays["sell_price"],
        carbon_price=arrays["carbon_price"],
        carbon_intensity=arrays["carbon_intensity"],
        tie_line_limit=float(config["tie_line_limit_kw"]),
        power_limit=float(config["ess"]["rated_power_kw"]),
        energy_capacity=float(config["ess"]["energy_capacity_kwh"]),
        desired_ess_power=shared_ess_signal * float(config["ess"]["rated_power_kw"]) * 0.22,
        carbon_multiplier=float(1.00 + 0.40 * llm_intent["operator_carbon_focus"].mean() - 0.06 * llm_intent["operator_econ_focus"].mean()),
        projection_weight=8.0,
    )
    return {"hvac": hvac, "service": service, "ev": ev, "ess": ess}


def _intent_array(values: Any, horizon: int, low: float, high: float, fallback: float | np.ndarray) -> np.ndarray:
    fallback_array = np.full(horizon, float(fallback), dtype=float) if np.isscalar(fallback) else np.asarray(fallback, dtype=float)
    if not isinstance(values, list) or len(values) != horizon:
        return fallback_array.copy()
    return np.clip(np.asarray(values, dtype=float), low, high)


def _restore_case1_intent(intent_payload: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    horizon = len(base_arrays(scenario)["hours"])
    return {
        "operator_econ_focus": _intent_array(intent_payload.get("operator_econ_focus"), horizon, 0.0, 1.0, 0.5),
        "operator_carbon_focus": _intent_array(intent_payload.get("operator_carbon_focus"), horizon, 0.0, 1.0, 0.5),
        "ess_signal": _intent_array(intent_payload.get("ess_signal"), horizon, -1.0, 1.0, 0.0),
        "hvac_signal": _intent_array(intent_payload.get("hvac_signal"), horizon, -1.0, 1.0, 0.0),
        "service_signal": _intent_array(intent_payload.get("service_signal"), horizon, -1.0, 1.0, 0.0),
        "ev_signal": _intent_array(intent_payload.get("ev_signal"), horizon, 0.0, 1.0, 0.0),
        "summary": intent_payload.get("summary", []),
        "agent_rationales": intent_payload.get("agent_rationales", {}),
        "model_name": intent_payload.get("model_name", "restored_trace"),
    }


def _neutral_profile_intent(intent: dict[str, Any]) -> dict[str, Any]:
    horizon = len(intent["operator_econ_focus"])
    neutral = dict(intent)
    neutral.update(
        {
            "operator_econ_focus": np.full(horizon, 0.5, dtype=float),
            "operator_carbon_focus": np.full(horizon, 0.5, dtype=float),
            "ess_signal": np.zeros(horizon, dtype=float),
            "hvac_signal": np.zeros(horizon, dtype=float),
            "service_signal": np.zeros(horizon, dtype=float),
            "ev_signal": np.full(horizon, float(np.mean(intent["ev_signal"])), dtype=float),
            "model_name": "neutral_profile_control",
        }
    )
    return neutral


def _operator_only_intent(intent: dict[str, Any]) -> dict[str, Any]:
    horizon = len(intent["operator_econ_focus"])
    operator_only = dict(intent)
    operator_only.update(
        {
            "ess_signal": np.zeros(horizon, dtype=float),
            "hvac_signal": np.zeros(horizon, dtype=float),
            "service_signal": np.zeros(horizon, dtype=float),
            "ev_signal": np.full(horizon, float(np.mean(intent["ev_signal"])), dtype=float),
            "summary": ["Resource-agent feedback signals are disabled; only operator-level preferences are retained."],
            "model_name": "operator_only_control",
        }
    )
    return operator_only


def _case1_ablation_runs(scenario: dict[str, Any], llm_intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mock_orchestrator = Case1MultiAgentOrchestrator(model="parameterized_preference_agent", use_mock_llm=True)
    parameterized_intents = [
        mock_orchestrator.generate_intent(scenario, idx, LLM_DIR / f"parameterized_ablation_run_{idx}.json")
        for idx in range(1, len(llm_intents) + 1)
    ]

    full_runs: list[BaselineRun] = []
    neutral_runs: list[BaselineRun] = []
    parameterized_runs: list[BaselineRun] = []
    operator_only_runs: list[BaselineRun] = []
    single_layer_runs: list[BaselineRun] = []

    for idx, intent in enumerate(llm_intents, start=1):
        full_runs.append(simulate_internal_plan("A1_Full_PA_MA_LLMs", f"a1_{idx}", scenario, _build_pi_plan(scenario, intent), intent, True))
        neutral_intent = _neutral_profile_intent(intent)
        neutral_runs.append(simulate_internal_plan("A2_No_Heterogeneous_Profile", f"a2_{idx}", scenario, _build_pi_plan(scenario, neutral_intent), neutral_intent, True))
        param_intent = parameterized_intents[idx - 1]
        parameterized_runs.append(simulate_internal_plan("A3_Parameterized_Agents", f"a3_{idx}", scenario, _build_pi_plan(scenario, param_intent), param_intent, True))
        operator_only_intent = _operator_only_intent(intent)
        operator_only_runs.append(simulate_internal_plan("A4_No_Resource_Agent_Feedback", f"a4_{idx}", scenario, _build_pi_plan(scenario, operator_only_intent), operator_only_intent, True))
        single_layer_runs.append(simulate_internal_plan("A5_Single_Layer_Manager", f"a5_{idx}", scenario, _build_single_llm_plan(scenario, intent), intent, True))

    return [
        aggregate_runs("A1_Full_PA_MA_LLMs", full_runs),
        aggregate_runs("A2_No_Heterogeneous_Profile", neutral_runs),
        aggregate_runs("A3_Parameterized_Agents", parameterized_runs),
        aggregate_runs("A4_No_Resource_Agent_Feedback", operator_only_runs),
        aggregate_runs("A5_Single_Layer_Manager", single_layer_runs),
    ]


def _save_case1_ablation_table(ablation_results: list[dict[str, Any]]) -> None:
    headers = [
        "Variant",
        "Total cost",
        "Carbon emission",
        "High-carbon purchase",
        "Fairness index",
        "Violation rate",
        "Cost std",
    ]
    with (CASE1_OUTPUT_DIR / "table3_case1_ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for item in ablation_results:
            metrics = item["metrics"]
            writer.writerow(
                [
                    item["baseline"],
                    metrics["total_operating_cost"]["mean"],
                    metrics["total_carbon_emission"]["mean"],
                    metrics["high_carbon_grid_purchase"]["mean"],
                    metrics["carbon_fairness_index"]["mean"],
                    metrics["constraint_violation_rate"]["mean"],
                    metrics["total_operating_cost"]["std"],
                ]
            )
    write_markdown(
        CASE1_OUTPUT_DIR / "table3_case1_ablation.md",
        "\n".join(
            ["# Table 3. Case I ablation controls", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + [
                "| "
                + " | ".join(
                    [
                        str(item["baseline"]),
                        str(item["metrics"]["total_operating_cost"]["mean"]),
                        str(item["metrics"]["total_carbon_emission"]["mean"]),
                        str(item["metrics"]["high_carbon_grid_purchase"]["mean"]),
                        str(item["metrics"]["carbon_fairness_index"]["mean"]),
                        str(item["metrics"]["constraint_violation_rate"]["mean"]),
                        str(item["metrics"]["total_operating_cost"]["std"]),
                    ]
                )
                + " |"
                for item in ablation_results
            ]
        ),
    )


def _save_case1_tables(scenario: dict[str, Any], aggregated: list[dict[str, Any]]) -> None:
    config = scenario["config"]
    rows = [
        ("Scheduling horizon (h)", config["horizon_hours"]),
        ("Tie-line limit (kW)", config["tie_line_limit_kw"]),
        ("PV rated power (kW)", config["pv"]["rated_power_kw"]),
        ("ESS rated power (kW)", config["ess"]["rated_power_kw"]),
        ("ESS energy capacity (kWh)", config["ess"]["energy_capacity_kwh"]),
        ("Inflexible peak load (kW)", config["inflexible_peak_kw"]),
        ("HVAC peak load (kW)", config["flexible_loads"][0]["baseline_peak_kw"]),
        ("HVAC daily energy (kWh/day)", config["flexible_loads"][0]["daily_energy_kwh"]),
        ("Service peak load (kW)", config["flexible_loads"][1]["baseline_peak_kw"]),
        ("Service daily energy (kWh/day)", config["flexible_loads"][1]["daily_energy_kwh"]),
        ("EV charging slots", config["ev_cluster"]["slots"]),
        ("EV max charging power (kW)", config["ev_cluster"]["max_charging_power_kw"]),
        ("EV typical daily energy (kWh/day)", config["ev_cluster"]["daily_energy_kwh"]),
    ]
    with (CASE1_OUTPUT_DIR / "table1_case1_parameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Parameter", "Value"])
        writer.writerows(rows)
    write_markdown(
        CASE1_OUTPUT_DIR / "table1_case1_parameters.md",
        "\n".join(
            ["# Table 1. Weekly low-carbon park parameters", "", "| Parameter | Value |", "|---|---:|"]
            + [f"| {name} | {value} |" for name, value in rows]
        ),
    )

    headers = [
        "Baseline",
        "Total cost",
        "Purchase cost",
        "Carbon cost",
        "Carbon emission",
        "Renewable utilization",
        "High-carbon purchase",
        "Fairness index",
        "Violation rate",
        "Cost std",
    ]
    with (CASE1_OUTPUT_DIR / "table2_case1_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for item in aggregated:
            metrics = item["metrics"]
            writer.writerow(
                [
                    item["baseline"],
                    metrics["total_operating_cost"]["mean"],
                    metrics["electricity_purchase_cost"]["mean"],
                    metrics["carbon_related_cost"]["mean"],
                    metrics["total_carbon_emission"]["mean"],
                    metrics["renewable_utilization"]["mean"],
                    metrics["high_carbon_grid_purchase"]["mean"],
                    metrics["carbon_fairness_index"]["mean"],
                    metrics["constraint_violation_rate"]["mean"],
                    metrics["total_operating_cost"]["std"],
                ]
            )
    write_markdown(
        CASE1_OUTPUT_DIR / "table2_case1_performance.md",
        "\n".join(
            ["# Table 2. Weekly Case I performance", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + [
                "| "
                + " | ".join(
                    [
                        str(item["baseline"]),
                        str(item["metrics"]["total_operating_cost"]["mean"]),
                        str(item["metrics"]["electricity_purchase_cost"]["mean"]),
                        str(item["metrics"]["carbon_related_cost"]["mean"]),
                        str(item["metrics"]["total_carbon_emission"]["mean"]),
                        str(item["metrics"]["renewable_utilization"]["mean"]),
                        str(item["metrics"]["high_carbon_grid_purchase"]["mean"]),
                        str(item["metrics"]["carbon_fairness_index"]["mean"]),
                        str(item["metrics"]["constraint_violation_rate"]["mean"]),
                        str(item["metrics"]["total_operating_cost"]["std"]),
                    ]
                )
                + " |"
                for item in aggregated
            ]
        ),
    )


def _save_case1_summary(aggregated: list[dict[str, Any]], reference_pi: BaselineRun) -> None:
    metrics_map = {item["baseline"]: item["metrics"] for item in aggregated}
    b1 = metrics_map["B1_Rule_Based_EMS"]
    b2 = metrics_map["B2_Centralized_Optimization"]
    b3 = metrics_map["B3_Single_LLM_Unified_Manager"]
    b4 = metrics_map["B4_LLM_MAS_wo_Physics"]
    b5 = metrics_map["B5_PA_MA_LLMs"]
    write_markdown(
        CASE1_OUTPUT_DIR / "case1_summary.md",
        "\n".join(
            [
                "# Case I Summary",
                "",
                "## Key findings",
                "",
                "- The 7x24 weekly PA-MA-LLMs baseline preserves feasibility while coordinating inter-day carbon-aware flexibility.",
                "- Relative to the rule-based EMS, the proposed method reduces weekly operating cost and high-carbon grid exposure.",
                "- Relative to the single-LLM unified manager, explicit multi-agent generation improves differentiated use of HVAC, service load, EV, and ESS flexibility.",
                "- Removing the physics-aware projection layer still leads to large infeasibility and unstable weekly outcomes.",
                "",
                "## Selected metrics",
                "",
                f"- `B1` total cost: {b1['total_operating_cost']['mean']:.3f} RMB",
                f"- `B2` total cost: {b2['total_operating_cost']['mean']:.3f} RMB",
                f"- `B3` total cost: {b3['total_operating_cost']['mean']:.3f} RMB",
                f"- `B4` total cost: {b4['total_operating_cost']['mean']:.3f} RMB",
                f"- `B5` total cost: {b5['total_operating_cost']['mean']:.3f} RMB",
                f"- `B4` violation rate: {b4['constraint_violation_rate']['mean']:.4f}",
                f"- `B5` violation rate: {b5['constraint_violation_rate']['mean']:.4f}",
                f"- `B5` fairness index: {b5['carbon_fairness_index']['mean']:.4f}",
                "",
                "## Reference PA-MA-LLMs run",
                "",
                f"- Run id: `{reference_pi.run_id}`",
                f"- Total cost: {reference_pi.total_operating_cost:.3f} RMB",
                f"- Carbon emission: {reference_pi.total_carbon_emission:.3f} kg",
                f"- Renewable utilization: {reference_pi.renewable_utilization * 100:.2f}%",
                f"- LLM summary: {', '.join(reference_pi.details.get('llm_summary', []))}",
            ]
        ),
    )


def run_case1(repeats: int = 3, model: str = DEFAULT_MODEL, use_mock_llm: bool = False) -> dict[str, Any]:
    scenario = _load_or_build_scenario()
    CASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    api_key = None if use_mock_llm else read_api_key(KEY_PATH)
    orchestrator = Case1MultiAgentOrchestrator(model=model, use_mock_llm=use_mock_llm, api_key=api_key)

    deterministic_runs = [
        simulate_internal_plan("B1_Rule_Based_EMS", "rule_1", scenario, _build_rule_based_plan(scenario), None, True),
        simulate_internal_plan("B2_Centralized_Optimization", "central_1", scenario, _build_centralized_plan(scenario), None, True),
    ]

    single_llm_runs: list[BaselineRun] = []
    llm_plain_runs: list[BaselineRun] = []
    pi_runs: list[BaselineRun] = []
    llm_intents: list[dict[str, Any]] = []
    llm_intent_records: list[dict[str, Any]] = []
    for idx in range(1, repeats + 1):
        trace_path = LLM_DIR / (f"mock_run_{idx}.json" if use_mock_llm else f"deepseek_run_{idx}.json")
        intent = orchestrator.generate_intent(scenario, idx, trace_path)
        llm_intents.append(intent)
        llm_intent_records.append(
            {
                "run_id": f"llm_{idx}",
                "intent": {
                    key: series_payload(value, digits=4) if isinstance(value, np.ndarray) else value
                    for key, value in intent.items()
                },
            }
        )
        single_llm_runs.append(simulate_internal_plan("B3_Single_LLM_Unified_Manager", f"single_{idx}", scenario, _build_single_llm_plan(scenario, intent), intent, True))
        llm_plain_runs.append(simulate_internal_plan("B4_LLM_MAS_wo_Physics", f"llm_{idx}", scenario, _build_raw_llm_plan(scenario, intent), intent, False))
        pi_runs.append(simulate_internal_plan("B5_PA_MA_LLMs", f"pa_{idx}", scenario, _build_pi_plan(scenario, intent), intent, True))

    aggregated = [
        aggregate_runs("B1_Rule_Based_EMS", [deterministic_runs[0]]),
        aggregate_runs("B2_Centralized_Optimization", [deterministic_runs[1]]),
        aggregate_runs("B3_Single_LLM_Unified_Manager", single_llm_runs),
        aggregate_runs("B4_LLM_MAS_wo_Physics", llm_plain_runs),
        aggregate_runs("B5_PA_MA_LLMs", pi_runs),
    ]
    reference_pi = select_reference_run(pi_runs)
    ablation_results = _case1_ablation_runs(scenario, llm_intents)

    _save_case1_tables(scenario, aggregated)
    _save_case1_ablation_table(ablation_results)
    plot_case1_dispatch(reference_pi, scenario, FIGURE_DIR)
    _save_case1_summary(aggregated, reference_pi)

    output_payload = {
        "case": "5.1 Case I: Internal Decision-Making in a Single Low-Carbon Park",
        "llm_model": model if not use_mock_llm else "mock_llm_for_smoke_test",
        "repeats": repeats,
        "aggregated_results": aggregated,
        "ablation_results": ablation_results,
        "reference_pa_run": reference_pi.__dict__,
        "llm_intents": llm_intent_records,
    }
    write_json(CASE1_OUTPUT_DIR / "case1_results.json", output_payload)
    return output_payload


def run_case1_ablation_from_existing() -> list[dict[str, Any]]:
    scenario = _load_or_build_scenario()
    payload_path = CASE1_OUTPUT_DIR / "case1_results.json"
    if not payload_path.exists():
        raise FileNotFoundError("Run Case I first so the LLM intent traces are available.")

    payload = load_json(payload_path)
    llm_intents = [_restore_case1_intent(record["intent"], scenario) for record in payload.get("llm_intents", [])]
    if not llm_intents:
        raise ValueError("No LLM intents found in existing Case I results.")

    ablation_results = _case1_ablation_runs(scenario, llm_intents)
    _save_case1_ablation_table(ablation_results)
    payload["ablation_results"] = ablation_results
    write_json(payload_path, payload)
    return ablation_results
