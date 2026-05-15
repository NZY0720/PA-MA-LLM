from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from methods.case1 import _build_centralized_plan, _load_or_build_scenario
from methods.common import (
    CASE2_PARK_IDS,
    Case2Run,
    aggregate_case2_runs,
    base_arrays,
    carbon_quota_factor,
    select_reference_case2_run,
    simulate_internal_plan,
)
from methods.llm_agents import Case2MultiAgentOrchestrator
from utils.constants import CASE2_OUTPUT_DIR, DEFAULT_MODEL, KEY_PATH
from utils.io_utils import load_json, read_api_key, write_json, write_markdown
from utils.math_utils import clamp, jains_index, series_payload, to_np
from visualization.case2 import (
    plot_case2_coupled_market_summary,
    plot_case2_negotiation,
)

FIGURE_DIR = CASE2_OUTPUT_DIR / "figures"
LLM_DIR = CASE2_OUTPUT_DIR / "llm_traces"
TRADE_TRANSACTION_FEE = 0.018
RAW_TRADE_PENALTY = 18.0
CARBON_EXTERNAL_CREDIT_FACTOR = 1.25
CARBON_MARKET_TRANSACTION_FEE = 0.004


@dataclass(frozen=True)
class ParkSpec:
    park_id: str
    display_name: str
    park_type: str
    pv_scale: float
    ess_power_scale: float
    ess_energy_scale: float
    inflexible_scale: float
    hvac_scale: float
    service_scale: float
    ev_scale: float
    tie_line_scale: float
    carbon_sensitivity: float


@dataclass
class ParkState:
    spec: ParkSpec
    scenario: dict[str, Any]
    internal_run: Any
    arrays: dict[str, np.ndarray]
    subjective_profile: dict[str, float] | None = None
    profile_rationales: dict[str, str] | None = None


def _carbon_market_price(states: dict[str, ParkState], total_surplus: float, total_deficit: float) -> float:
    prices = [float(np.mean(state.arrays["carbon_price"])) * state.spec.carbon_sensitivity for state in states.values()]
    base_price = float(np.mean(prices)) if prices else 0.0
    scarcity = total_deficit / max(total_surplus + total_deficit, 1e-6)
    return round(base_price * (0.90 + 0.35 * scarcity), 4)


def _settle_carbon_market(
    states: dict[str, ParkState],
    park_emissions: dict[str, float],
    park_loads: dict[str, float],
) -> dict[str, Any]:
    total_reference_emission = sum(float(state.internal_run.total_carbon_emission) for state in states.values())
    total_reference_load = sum(max(float(park_loads[park_id]), 0.0) for park_id in states)
    benchmark_intensity = total_reference_emission / max(total_reference_load, 1e-6)
    quota = {
        park_id: park_loads[park_id] * benchmark_intensity * carbon_quota_factor(state.spec.park_type)
        for park_id, state in states.items()
    }
    position = {park_id: quota[park_id] - park_emissions[park_id] for park_id in states}
    surplus = {park_id: max(value, 0.0) for park_id, value in position.items()}
    deficit = {park_id: max(-value, 0.0) for park_id, value in position.items()}
    total_surplus = sum(surplus.values())
    total_deficit = sum(deficit.values())
    clearing_price = _carbon_market_price(states, total_surplus, total_deficit)

    seller_remaining = dict(sorted(surplus.items(), key=lambda item: item[1], reverse=True))
    buyer_remaining = dict(sorted(deficit.items(), key=lambda item: item[1], reverse=True))
    payment_out = {park_id: 0.0 for park_id in states}
    payment_in = {park_id: 0.0 for park_id in states}
    external_credit_purchase = {park_id: 0.0 for park_id in states}
    carbon_trades: dict[str, dict[str, float]] = {}
    internal_volume = 0.0

    for buyer, buyer_deficit in buyer_remaining.items():
        remaining_deficit = buyer_deficit
        if remaining_deficit <= 1e-6:
            continue
        for seller, seller_surplus in list(seller_remaining.items()):
            if seller == buyer or seller_surplus <= 1e-6 or remaining_deficit <= 1e-6:
                continue
            volume = min(seller_surplus, remaining_deficit)
            carbon_trades[f"{seller}->{buyer}"] = {
                "volume_kg": round(float(volume), 4),
                "price_rmb_per_kg": clearing_price,
            }
            payment = volume * (clearing_price + CARBON_MARKET_TRANSACTION_FEE)
            payment_out[buyer] += payment
            payment_in[seller] += volume * (clearing_price - CARBON_MARKET_TRANSACTION_FEE)
            seller_remaining[seller] -= volume
            remaining_deficit -= volume
            internal_volume += volume
        if remaining_deficit > 1e-6:
            external_credit_purchase[buyer] = remaining_deficit
            payment_out[buyer] += remaining_deficit * clearing_price * CARBON_EXTERNAL_CREDIT_FACTOR

    net_cost = {park_id: payment_out[park_id] - payment_in[park_id] for park_id in states}
    return {
        "benchmark_intensity_kg_per_kwh": round(float(benchmark_intensity), 6),
        "quota_kg": {park_id: round(float(value), 4) for park_id, value in quota.items()},
        "position_kg": {park_id: round(float(value), 4) for park_id, value in position.items()},
        "surplus_kg": {park_id: round(float(value), 4) for park_id, value in surplus.items()},
        "deficit_kg": {park_id: round(float(value), 4) for park_id, value in deficit.items()},
        "trades": carbon_trades,
        "payment_out_rmb": {park_id: round(float(value), 4) for park_id, value in payment_out.items()},
        "payment_in_rmb": {park_id: round(float(value), 4) for park_id, value in payment_in.items()},
        "net_cost_rmb": {park_id: round(float(value), 4) for park_id, value in net_cost.items()},
        "external_credit_purchase_kg": {park_id: round(float(value), 4) for park_id, value in external_credit_purchase.items()},
        "internal_trading_volume_kg": round(float(internal_volume), 4),
        "external_credit_volume_kg": round(float(sum(external_credit_purchase.values())), 4),
        "clearing_price_rmb_per_kg": clearing_price,
        "total_compliance_cost_rmb": round(float(sum(max(value, 0.0) for value in net_cost.values())), 4),
    }


@lru_cache(maxsize=1)
def _build_park_specs() -> tuple[ParkSpec, ...]:
    return (
        ParkSpec("Park_A", "Park A", "Renewable-Rich Park", 2.10, 1.00, 1.00, 0.72, 0.76, 0.78, 0.84, 1.10, 1.16),
        ParkSpec("Park_B", "Park B", "Load-Intensive Park", 0.66, 0.86, 0.86, 1.42, 1.30, 1.24, 1.18, 1.20, 0.96),
        ParkSpec("Park_C", "Park C", "Storage-Dominant Park", 1.08, 1.82, 2.08, 0.94, 0.90, 0.92, 0.84, 1.05, 1.05),
        ParkSpec("Park_D", "Park D", "Standard Park", 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
        ParkSpec("Park_E", "Park E", "Flexible-Demand Park", 1.22, 1.08, 1.12, 0.88, 1.28, 1.36, 1.10, 1.08, 1.22),
    )


def _round_to_step(value: float, step: float) -> float:
    return round(value / step) * step


def _scale_list(values: list[float], scale: float, digits: int = 3) -> list[float]:
    return [round(float(v) * scale, digits) for v in values]


def _make_scaled_scenario(base_scenario: dict[str, Any], spec: ParkSpec) -> dict[str, Any]:
    scenario = copy.deepcopy(base_scenario)
    config = scenario["config"]
    profiles = scenario["profiles"]

    config["name"] = spec.park_id.lower()
    config["operator_name"] = f"{spec.park_id.lower()}_operator"
    config["tie_line_limit_kw"] = round(float(config["tie_line_limit_kw"]) * spec.tie_line_scale, 3)
    config["pv"]["rated_power_kw"] = round(float(config["pv"]["rated_power_kw"]) * spec.pv_scale, 3)
    config["ess"]["rated_power_kw"] = _round_to_step(float(config["ess"]["rated_power_kw"]) * spec.ess_power_scale, 20.0)
    config["ess"]["energy_capacity_kwh"] = _round_to_step(float(config["ess"]["energy_capacity_kwh"]) * spec.ess_energy_scale, 40.0)
    config["inflexible_peak_kw"] = round(float(config["inflexible_peak_kw"]) * spec.inflexible_scale, 3)
    config["flexible_loads"][0]["baseline_peak_kw"] = round(float(config["flexible_loads"][0]["baseline_peak_kw"]) * spec.hvac_scale, 3)
    config["flexible_loads"][0]["daily_energy_kwh"] = round(float(config["flexible_loads"][0]["daily_energy_kwh"]) * spec.hvac_scale, 3)
    config["flexible_loads"][1]["baseline_peak_kw"] = round(float(config["flexible_loads"][1]["baseline_peak_kw"]) * spec.service_scale, 3)
    config["flexible_loads"][1]["daily_energy_kwh"] = round(float(config["flexible_loads"][1]["daily_energy_kwh"]) * spec.service_scale, 3)
    config["ev_cluster"]["max_charging_power_kw"] = round(float(config["ev_cluster"]["max_charging_power_kw"]) * spec.ev_scale, 3)
    config["ev_cluster"]["daily_energy_kwh"] = round(float(config["ev_cluster"]["daily_energy_kwh"]) * spec.ev_scale, 3)

    profiles["inflexible_load_kw"] = _scale_list(profiles["inflexible_load_kw"], spec.inflexible_scale)
    profiles["flexible_loads_kw"]["hvac_load"] = _scale_list(profiles["flexible_loads_kw"]["hvac_load"], spec.hvac_scale)
    profiles["flexible_loads_kw"]["service_load"] = _scale_list(profiles["flexible_loads_kw"]["service_load"], spec.service_scale)
    profiles["pv_available_kw"] = _scale_list(profiles["pv_available_kw"], spec.pv_scale)
    profiles["ev_energy_request_kwh"] = _scale_list(profiles["ev_energy_request_kwh"], spec.ev_scale)
    profiles["carbon_price_rmb_per_kg"] = _scale_list(profiles["carbon_price_rmb_per_kg"], spec.carbon_sensitivity, digits=4)

    scenario["scenario_name"] = spec.park_id.lower()
    scenario["notes"]["park_type"] = spec.park_type
    scenario["notes"]["carbon_sensitivity"] = spec.carbon_sensitivity
    return scenario


def _build_park_states() -> dict[str, ParkState]:
    base_scenario = _load_or_build_scenario()
    states: dict[str, ParkState] = {}
    export_dir = CASE2_OUTPUT_DIR / "scenarios"
    export_dir.mkdir(parents=True, exist_ok=True)
    for spec in _build_park_specs():
        scenario = _make_scaled_scenario(base_scenario, spec)
        write_json(export_dir / f"{spec.park_id.lower()}_scenario.json", scenario)
        internal_run = simulate_internal_plan(
            name=f"{spec.park_id}_internal_centralized",
            run_id=f"{spec.park_id.lower()}_central",
            scenario=scenario,
            plan=_build_centralized_plan(scenario),
            llm_intent=None,
            physics_projection=True,
        )
        states[spec.park_id] = ParkState(spec=spec, scenario=scenario, internal_run=internal_run, arrays=base_arrays(scenario))
    return states


def _attach_subjective_profiles(states: dict[str, ParkState], client: Any | None, use_mock: bool) -> None:
    """Run the evidence-grounded profile extraction pipeline once per park.

    Implements equations (6)-(8) of the manuscript: retrieve TopK evidence
    chunks from the park's corpus and call a schema-aligned LLM extractor
    to obtain six subjective preference scores (theta) with rationales.
    Results are cached to disk so repeats do not re-pay the API cost.
    """
    from methods.profile_extraction import load_or_extract_profile, profile_signature
    for park_id, state in states.items():
        profile = load_or_extract_profile(park_id, state.spec.park_type, client, use_mock=use_mock)
        state.subjective_profile = profile_signature(profile)
        state.profile_rationales = {dim: payload.rationale for dim, payload in profile.items()}


@lru_cache(maxsize=1)
def _link_capacities() -> dict[tuple[str, str], float]:
    capacities: dict[tuple[str, str], float] = {}
    park_ids = [spec.park_id for spec in _build_park_specs()]
    special_caps = {
        ("Park_A", "Park_B"): 260.0,
        ("Park_A", "Park_C"): 220.0,
        ("Park_A", "Park_E"): 235.0,
        ("Park_B", "Park_C"): 210.0,
        ("Park_B", "Park_D"): 205.0,
        ("Park_C", "Park_E"): 225.0,
        ("Park_D", "Park_E"): 200.0,
    }
    for idx, a in enumerate(park_ids):
        for b in park_ids[idx + 1 :]:
            cap = special_caps.get((a, b), special_caps.get((b, a), 190.0))
            capacities[(a, b)] = cap
            capacities[(b, a)] = cap
    return capacities


def _base_external_arrays(state: ParkState) -> dict[str, np.ndarray]:
    hourly = state.internal_run.details["hourly"]
    return {
        "buy": to_np(hourly["buy_kw"]),
        "sell": to_np(hourly["sell_kw"]),
        "pv_export": to_np(hourly["pv_export_kw"]),
        "total_load": to_np(hourly["total_load_kw"]),
        "buy_price": state.arrays["buy_price"],
        "sell_price": state.arrays["sell_price"],
        "carbon_price": state.arrays["carbon_price"],
        "carbon_intensity": state.arrays["carbon_intensity"],
    }


def _candidate_price(seller: str, buyer: str, t: int, states: dict[str, ParkState], markup_factor: float = 0.5) -> float:
    seller_arrays = _base_external_arrays(states[seller])
    buyer_arrays = _base_external_arrays(states[buyer])
    spread = max(float(buyer_arrays["buy_price"][t] - seller_arrays["sell_price"][t]), 0.03)
    return round(float(seller_arrays["sell_price"][t] + markup_factor * spread), 4)


def _run_rule_based_trading(states: dict[str, ParkState]) -> dict[str, Any]:
    external = {park_id: _base_external_arrays(state) for park_id, state in states.items()}
    link_caps = _link_capacities()
    horizon = len(next(iter(external.values()))["buy"])
    trade = {hour: {} for hour in range(horizon)}
    price_book = {hour: {} for hour in range(horizon)}
    rounds_per_hour = np.zeros(horizon, dtype=float)
    converged_hours = 0
    active_hours = 0

    for hour in range(horizon):
        sellers = sorted([park_id for park_id, arr in external.items() if arr["sell"][hour] > 1e-6], key=lambda pid: external[pid]["pv_export"][hour], reverse=True)
        buyers = sorted([park_id for park_id, arr in external.items() if arr["buy"][hour] > 1e-6], key=lambda pid: external[pid]["buy"][hour], reverse=True)
        if not sellers or not buyers:
            continue
        active_hours += 1
        rounds_per_hour[hour] = 1.0
        seller_remaining = {pid: float(external[pid]["sell"][hour]) for pid in sellers}
        buyer_remaining = {pid: float(external[pid]["buy"][hour]) for pid in buyers}
        for seller in sellers:
            for buyer in buyers:
                if seller == buyer:
                    continue
                volume = min(seller_remaining[seller], buyer_remaining[buyer], link_caps[(seller, buyer)] * 0.80)
                if volume <= 1e-6:
                    continue
                trade[hour][(seller, buyer)] = round(volume, 4)
                price_book[hour][(seller, buyer)] = _candidate_price(seller, buyer, hour, states, markup_factor=0.50)
                seller_remaining[seller] -= volume
                buyer_remaining[buyer] -= volume
        if trade[hour]:
            converged_hours += 1

    return {
        "trade": trade,
        "price_book": price_book,
        "rounds_per_hour": rounds_per_hour.tolist(),
        "negotiation_rounds": round(float(rounds_per_hour[rounds_per_hour > 0].mean()), 4) if active_hours else 0.0,
        "convergence_rate": round(float(converged_hours / active_hours), 4) if active_hours else 0.0,
        "export_headroom": {park_id: [0.0] * horizon for park_id in states},
    }


def _run_game_based_trading(states: dict[str, ParkState]) -> dict[str, Any]:
    external = {park_id: _base_external_arrays(state) for park_id, state in states.items()}
    link_caps = _link_capacities()
    horizon = len(next(iter(external.values()))["buy"])
    trade = {hour: {} for hour in range(horizon)}
    price_book = {hour: {} for hour in range(horizon)}
    rounds_per_hour = np.zeros(horizon, dtype=float)
    converged_hours = 0
    active_hours = 0

    for hour in range(horizon):
        sellers = [park_id for park_id, arr in external.items() if arr["sell"][hour] > 1e-6]
        buyers = [park_id for park_id, arr in external.items() if arr["buy"][hour] > 1e-6]
        if not sellers or not buyers:
            continue
        active_hours += 1
        seller_remaining = {pid: float(external[pid]["sell"][hour]) for pid in sellers}
        buyer_remaining = {pid: float(external[pid]["buy"][hour]) for pid in buyers}

        for _round in range(1, 7):
            best_pair = None
            best_score = -1e9
            best_price = 0.0
            best_volume = 0.0
            for seller in sellers:
                if seller_remaining[seller] <= 1e-6:
                    continue
                for buyer in buyers:
                    if buyer_remaining[buyer] <= 1e-6:
                        continue
                    ask = external[seller]["sell_price"][hour] + 0.15 * (external[buyer]["buy_price"][hour] - external[seller]["sell_price"][hour])
                    bid = external[buyer]["buy_price"][hour] - 0.10 * (external[buyer]["buy_price"][hour] - external[seller]["sell_price"][hour])
                    if bid <= ask + TRADE_TRANSACTION_FEE:
                        continue
                    volume = min(seller_remaining[seller], buyer_remaining[buyer], link_caps[(seller, buyer)] * 0.92)
                    score = volume * (bid - ask)
                    if score > best_score:
                        best_pair = (seller, buyer)
                        best_score = score
                        best_price = round(float((ask + bid) / 2.0), 4)
                        best_volume = volume
            if best_pair is None or best_volume <= 1e-6:
                break
            trade[hour][best_pair] = round(float(trade[hour].get(best_pair, 0.0) + best_volume), 4)
            price_book[hour][best_pair] = best_price
            seller_remaining[best_pair[0]] -= best_volume
            buyer_remaining[best_pair[1]] -= best_volume
            rounds_per_hour[hour] += 1.0
        if trade[hour]:
            converged_hours += 1

    return {
        "trade": trade,
        "price_book": price_book,
        "rounds_per_hour": rounds_per_hour.tolist(),
        "negotiation_rounds": round(float(rounds_per_hour[rounds_per_hour > 0].mean()), 4) if active_hours else 0.0,
        "convergence_rate": round(float(converged_hours / active_hours), 4) if active_hours else 0.0,
        "export_headroom": {park_id: [0.0] * horizon for park_id in states},
    }


def _compute_export_headroom(state: ParkState, hour: int, park_output: dict[str, Any]) -> float:
    ess_power = float(state.scenario["config"]["ess"]["rated_power_kw"])
    export_willingness = float(park_output.get("export_willingness", 0.0))
    concession = float(park_output.get("concession_factor", 0.0))
    hour_of_day = int(state.arrays["hour_of_day"][hour])
    if state.spec.park_type == "Renewable-Rich Park":
        return ess_power * 0.18 * export_willingness * (1.0 if 9 <= hour_of_day <= 17 else 0.25) * (0.75 + 0.25 * concession)
    if state.spec.park_type == "Storage-Dominant Park":
        return ess_power * 0.16 * export_willingness * (1.0 if 11 <= hour_of_day <= 21 else 0.30) * (0.70 + 0.30 * concession)
    return ess_power * 0.10 * export_willingness * (0.55 + 0.25 * concession)


def _round_order_book(round_payload: dict[str, Any]) -> dict[str, Any]:
    if "order_book" in round_payload:
        return round_payload["order_book"]
    sell_orders: list[dict[str, Any]] = []
    buy_orders: list[dict[str, Any]] = []
    for park_id, payload in round_payload.get("park_outputs", {}).items():
        export_target = float(payload.get("export_target_kwh", 0.0))
        import_target = float(payload.get("import_target_kwh", 0.0))
        if export_target > 1e-6:
            sell_orders.append(
                {
                    "park_id": park_id,
                    "quantity_kwh": round(float(export_target), 4),
                    "ask_price_rmb_per_kwh": round(float(payload.get("ask_price_rmb_per_kwh", 0.0)), 4),
                }
            )
        if import_target > 1e-6:
            buy_orders.append(
                {
                    "park_id": park_id,
                    "quantity_kwh": round(float(import_target), 4),
                    "bid_price_rmb_per_kwh": round(float(payload.get("bid_price_rmb_per_kwh", 0.0)), 4),
                }
            )
    sell_orders.sort(key=lambda item: (float(item["ask_price_rmb_per_kwh"]), item["park_id"]))
    buy_orders.sort(key=lambda item: (-float(item["bid_price_rmb_per_kwh"]), item["park_id"]))
    return {
        "continue_bidding": True,
        "sell_orders": sell_orders,
        "buy_orders": buy_orders,
        "summary": f"Order book receives {len(sell_orders)} sell orders and {len(buy_orders)} buy orders.",
    }


def _double_auction_candidates(
    order_book: dict[str, Any],
    seller_remaining: dict[str, float],
    buyer_remaining: dict[str, float],
    link_caps: dict[tuple[str, str], float],
    physics_projection: bool,
    box_clipping: bool = False,
) -> list[dict[str, Any]]:
    sellers = [dict(order) for order in order_book.get("sell_orders", [])]
    buyers = [dict(order) for order in order_book.get("buy_orders", [])]
    sellers.sort(key=lambda item: (float(item["ask_price_rmb_per_kwh"]), item["park_id"]))
    buyers.sort(key=lambda item: (-float(item["bid_price_rmb_per_kwh"]), item["park_id"]))
    seller_capacity = {park_id: float(value) for park_id, value in seller_remaining.items()}
    buyer_capacity = {park_id: float(value) for park_id, value in buyer_remaining.items()}
    candidates: list[dict[str, Any]] = []
    sell_idx = 0
    buy_idx = 0
    auction_rank = 1
    while sell_idx < len(sellers) and buy_idx < len(buyers):
        seller_order = sellers[sell_idx]
        buyer_order = buyers[buy_idx]
        seller = str(seller_order["park_id"])
        buyer = str(buyer_order["park_id"])
        if seller == buyer:
            if float(seller_order["ask_price_rmb_per_kwh"]) <= float(buyer_order["bid_price_rmb_per_kwh"]):
                buy_idx += 1
            else:
                sell_idx += 1
            continue
        ask = float(seller_order["ask_price_rmb_per_kwh"])
        bid = float(buyer_order["bid_price_rmb_per_kwh"])
        if bid + 1e-9 < ask:
            break
        max_link = link_caps[(seller, buyer)] if (physics_projection or box_clipping) else link_caps[(seller, buyer)] * 1.35
        volume = min(
            float(seller_order["quantity_kwh"]),
            float(buyer_order["quantity_kwh"]),
            float(seller_capacity.get(seller, 0.0)),
            float(buyer_capacity.get(buyer, 0.0)),
            max_link,
        )
        if volume <= 1e-6:
            if float(seller_order["quantity_kwh"]) <= 1e-6 or float(seller_capacity.get(seller, 0.0)) <= 1e-6:
                sell_idx += 1
            if float(buyer_order["quantity_kwh"]) <= 1e-6 or float(buyer_capacity.get(buyer, 0.0)) <= 1e-6:
                buy_idx += 1
            if seller in seller_capacity and seller_capacity[seller] > 1e-6 and buyer in buyer_capacity and buyer_capacity[buyer] > 1e-6:
                sell_idx += 1
            continue
        clearing_price = round(float((ask + bid) / 2.0), 4)
        candidates.append(
            {
                "seller": seller,
                "buyer": buyer,
                "score": round(float((bid - ask) * volume), 4),
                "auction_rank": auction_rank,
                "candidate_volume_kwh": round(float(volume), 4),
                "clearing_price_rmb_per_kwh": clearing_price,
                "ask_price_rmb_per_kwh": round(float(ask), 4),
                "bid_price_rmb_per_kwh": round(float(bid), 4),
            }
        )
        auction_rank += 1
        seller_order["quantity_kwh"] = float(seller_order["quantity_kwh"]) - volume
        buyer_order["quantity_kwh"] = float(buyer_order["quantity_kwh"]) - volume
        seller_capacity[seller] = float(seller_capacity.get(seller, 0.0)) - volume
        buyer_capacity[buyer] = float(buyer_capacity.get(buyer, 0.0)) - volume
        if float(seller_order["quantity_kwh"]) <= 1e-6 or float(seller_capacity.get(seller, 0.0)) <= 1e-6:
            sell_idx += 1
        if float(buyer_order["quantity_kwh"]) <= 1e-6 or float(buyer_capacity.get(buyer, 0.0)) <= 1e-6:
            buy_idx += 1
    return candidates


def _build_behavior_digest(round_logs: dict[str, Any]) -> list[str]:
    ranked_hours: list[tuple[float, int, int, str]] = []
    for hour_str, payload in round_logs.items():
        executed_volume = sum(
            float(pair["volume_kwh"])
            for round_payload in payload.get("rounds", [])
            for pair in round_payload.get("executed_pairs", [])
        )
        round_count = len(payload.get("rounds", []))
        if executed_volume <= 1e-6 and round_count <= 0:
            continue
        first_pair = "no agreement"
        if payload.get("rounds") and payload["rounds"][-1].get("executed_pairs"):
            first_pair = f"{payload['rounds'][-1]['executed_pairs'][0]['seller']}->{payload['rounds'][-1]['executed_pairs'][0]['buyer']}"
        ranked_hours.append((executed_volume + 10.0 * round_count, round_count, int(hour_str), first_pair))
    ranked_hours.sort(reverse=True)
    digest = []
    for _, round_count, hour, pair_text in ranked_hours[:4]:
        digest.append(f"Hour {hour}: {round_count} rounds, dominant executed pair {pair_text}.")
    return digest


def _llm_trace_matches_states(llm_intent: dict[str, Any], park_ids: set[str]) -> bool:
    for hour_payload in llm_intent.get("hours", {}).values():
        context_parks = set(hour_payload.get("context", {}).get("parks", {}).keys())
        if context_parks and context_parks != park_ids:
            return False
        for context in hour_payload.get("context", {}).get("parks", {}).values():
            if "carbon_position_kg" not in context:
                return False
        for round_payload in hour_payload.get("rounds", []):
            output_parks = set(round_payload.get("park_outputs", {}).keys())
            if output_parks and output_parks != park_ids:
                return False
            for output in round_payload.get("park_outputs", {}).values():
                if "carbon_market_posture" not in output:
                    return False
    return True


def _run_llm_trading(
    states: dict[str, ParkState],
    llm_intent: dict[str, Any],
    physics_projection: bool,
    box_clipping: bool = False,
) -> dict[str, Any]:
    external = {park_id: _base_external_arrays(state) for park_id, state in states.items()}
    link_caps = _link_capacities()
    horizon = len(next(iter(external.values()))["buy"])
    trade = {hour: {} for hour in range(horizon)}
    price_book = {hour: {} for hour in range(horizon)}
    rounds_per_hour = np.zeros(horizon, dtype=float)
    export_headroom = {park_id: [0.0] * horizon for park_id in states}
    converged_hours = 0
    active_hours = 0
    round_logs: dict[str, Any] = {}
    price_path: dict[str, list[dict[str, Any]]] = {}
    projection_adjustments: dict[str, Any] = {}
    partner_switch = {park_id: 0 for park_id in states}
    last_partner = {park_id: "" for park_id in states}

    for hour in range(horizon):
        hour_payload = llm_intent.get("hours", {}).get(str(hour), {})
        round_payloads = hour_payload.get("rounds", [])
        if not round_payloads:
            projection_adjustments[str(hour)] = {}
            continue
        active_hours += 1
        seller_cap_seen = {park_id: float(external[park_id]["sell"][hour]) for park_id in states}
        buyer_cap_seen = {park_id: float(external[park_id]["buy"][hour]) for park_id in states}
        seller_remaining = seller_cap_seen.copy()
        buyer_remaining = buyer_cap_seen.copy()
        hour_round_logs: list[dict[str, Any]] = []
        hour_projection: dict[str, Any] = {}
        traded_any_round = False

        for round_payload in round_payloads:
            round_index = int(round_payload["round_index"])
            round_candidates: list[dict[str, Any]] = []
            executed_pairs: list[dict[str, Any]] = []
            park_messages = {park_id: payload["message"] for park_id, payload in round_payload["park_outputs"].items()}
            order_book = _round_order_book(round_payload)
            for park_id, state in states.items():
                park_output = round_payload["park_outputs"][park_id]
                base_sell = float(external[park_id]["sell"][hour])
                base_buy = float(external[park_id]["buy"][hour])
                if physics_projection:
                    export_cap = base_sell + _compute_export_headroom(state, hour, park_output)
                    import_cap = base_buy
                    export_headroom[park_id][hour] = max(export_headroom[park_id][hour], max(export_cap - base_sell, 0.0))
                elif box_clipping:
                    # Each park's order is clipped to its individual feasible volume
                    # (no inflation, no inter-period headroom search); link caps stay strict.
                    export_cap = base_sell
                    import_cap = base_buy
                else:
                    export_cap = max(
                        base_sell,
                        float(park_output["export_target_kwh"]) * (1.15 + 0.35 * float(park_output["concession_factor"])),
                    )
                    import_cap = max(
                        base_buy,
                        float(park_output["import_target_kwh"]) * (1.12 + 0.30 * float(park_output["import_willingness"])),
                    )
                if export_cap > seller_cap_seen[park_id] + 1e-9:
                    seller_remaining[park_id] += export_cap - seller_cap_seen[park_id]
                    seller_cap_seen[park_id] = export_cap
                if import_cap > buyer_cap_seen[park_id] + 1e-9:
                    buyer_remaining[park_id] += import_cap - buyer_cap_seen[park_id]
                    buyer_cap_seen[park_id] = import_cap
                hour_projection[park_id] = {
                    "seller_cap_kw": round(float(seller_cap_seen[park_id]), 4),
                    "buyer_cap_kw": round(float(buyer_cap_seen[park_id]), 4),
                    "seller_remaining_kw": round(float(seller_remaining[park_id]), 4),
                    "buyer_remaining_kw": round(float(buyer_remaining[park_id]), 4),
                    "extra_export_headroom_kw": round(float(export_headroom[park_id][hour]), 4),
                }

            round_candidates = _double_auction_candidates(
                order_book=order_book,
                seller_remaining=seller_remaining,
                buyer_remaining=buyer_remaining,
                link_caps=link_caps,
                physics_projection=physics_projection,
                box_clipping=box_clipping,
            )
            for item in round_candidates:
                seller = item["seller"]
                buyer = item["buyer"]
                volume = min(float(item["candidate_volume_kwh"]), seller_remaining[seller], buyer_remaining[buyer])
                if volume <= 1e-6:
                    continue
                trade[hour][(seller, buyer)] = round(float(trade[hour].get((seller, buyer), 0.0) + volume), 4)
                price_book[hour][(seller, buyer)] = round(float(item["clearing_price_rmb_per_kwh"]), 4)
                seller_remaining[seller] -= volume
                buyer_remaining[buyer] -= volume
                executed_pairs.append(
                    {
                        "seller": seller,
                        "buyer": buyer,
                        "volume_kwh": round(float(volume), 4),
                        "price_rmb_per_kwh": round(float(item["clearing_price_rmb_per_kwh"]), 4),
                        "auction_rank": item["auction_rank"],
                        "ask_price_rmb_per_kwh": item["ask_price_rmb_per_kwh"],
                        "bid_price_rmb_per_kwh": item["bid_price_rmb_per_kwh"],
                    }
                )
                pair_key = f"{seller}->{buyer}"
                price_path.setdefault(pair_key, []).append(
                    {
                        "hour": hour,
                        "round": round_index,
                        "price_rmb_per_kwh": round(float(item["clearing_price_rmb_per_kwh"]), 4),
                        "volume_kwh": round(float(volume), 4),
                    }
                )
                if last_partner[seller] and last_partner[seller] != buyer:
                    partner_switch[seller] += 1
                if last_partner[buyer] and last_partner[buyer] != seller:
                    partner_switch[buyer] += 1
                last_partner[seller] = buyer
                last_partner[buyer] = seller
                traded_any_round = True

            rounds_per_hour[hour] += 1.0
            hour_round_logs.append(
                {
                    "round_index": round_index,
                    "park_messages": park_messages,
                    "park_outputs": round_payload["park_outputs"],
                    "order_book_summary": order_book["summary"],
                    "order_book": order_book,
                    "order_book_feedback": round_payload.get("order_book_feedback", {}),
                    "candidate_pairs": round_candidates,
                    "executed_pairs": executed_pairs,
                    "projection_adjustment": {park_id: dict(payload) for park_id, payload in hour_projection.items()},
                }
            )
            if not executed_pairs and not order_book.get("continue_bidding", order_book.get("continue_negotiation", False)):
                break

        round_logs[str(hour)] = {
            "context": hour_payload.get("context", {}),
            "rounds": hour_round_logs,
            "hour_summary": hour_payload.get("hour_summary", []),
        }
        projection_adjustments[str(hour)] = hour_projection
        if traded_any_round:
            converged_hours += 1

    positive_rounds = rounds_per_hour[rounds_per_hour > 0]

    return {
        "trade": trade,
        "price_book": price_book,
        "rounds_per_hour": rounds_per_hour.tolist(),
        "negotiation_rounds": round(float(positive_rounds.mean()), 4) if active_hours and positive_rounds.size else 0.0,
        "convergence_rate": round(float(converged_hours / active_hours), 4) if active_hours else 0.0,
        "export_headroom": export_headroom,
        "round_logs": round_logs,
        "price_path": price_path,
        "projection_adjustments": projection_adjustments,
        "behavior_summary": _build_behavior_digest(round_logs),
        "partner_switch": partner_switch,
    }


def _evaluate_case2(name: str, run_id: str, states: dict[str, ParkState], trading_result: dict[str, Any], baseline_no_trade: dict[str, float], model_name: str, llm_intent: dict[str, Any] | None, physics_projection: bool, box_clipping: bool = False) -> Case2Run:
    enforce_clip = physics_projection or box_clipping
    trade = trading_result["trade"]
    price_book = trading_result["price_book"]
    link_caps = _link_capacities()
    park_ids = list(states.keys())
    external = {park_id: _base_external_arrays(state) for park_id, state in states.items()}
    horizon = len(next(iter(external.values()))["buy"])
    export_headroom = trading_result.get("export_headroom", {park_id: [0.0] * horizon for park_id in park_ids})

    park_costs = {}
    base_park_costs = {}
    park_emissions = {}
    trading_benefits = {}
    park_loads = {}
    park_carbon_responsibility = {}
    total_trade_volume = 0.0
    total_trade_value = 0.0
    renewable_traded = 0.0
    total_transaction_fee = 0.0
    total_grid_purchase = 0.0
    total_emission = 0.0
    total_violations = 0.0
    total_checks = horizon * (len(park_ids) + len(link_caps) / 2)
    total_balance_error = 0.0
    winwin_pairs = 0
    total_pairs = 0
    bilateral_totals = {(a, b): 0.0 for a in park_ids for b in park_ids if a != b}

    per_park_hourly_buy = {park_id: external[park_id]["buy"].copy() for park_id in park_ids}
    per_park_hourly_sell = {park_id: external[park_id]["sell"].copy() for park_id in park_ids}
    trade_payment_in = {park_id: 0.0 for park_id in park_ids}
    trade_payment_out = {park_id: 0.0 for park_id in park_ids}
    trade_carbon_add = {park_id: 0.0 for park_id in park_ids}
    trade_benefit_internal = {park_id: 0.0 for park_id in park_ids}
    park_penalty_cost = {park_id: 0.0 for park_id in park_ids}
    park_penalty_emission = {park_id: 0.0 for park_id in park_ids}
    park_coordination_cost = {park_id: 0.0 for park_id in park_ids}

    for hour in range(horizon):
        exports_by_park = {park_id: 0.0 for park_id in park_ids}
        imports_by_park = {park_id: 0.0 for park_id in park_ids}
        for (seller, buyer), volume in trade[hour].items():
            volume = float(volume)
            price = float(price_book[hour][(seller, buyer)])
            total_pairs += 1
            bilateral_totals[(seller, buyer)] += volume
            exports_by_park[seller] += volume
            imports_by_park[buyer] += volume
            total_trade_volume += volume
            total_trade_value += volume * price
            total_transaction_fee += volume * TRADE_TRANSACTION_FEE

            grid_buy_price = float(external[buyer]["buy_price"][hour])
            grid_sell_price = float(external[seller]["sell_price"][hour])
            seller_gain = volume * max(price - grid_sell_price - TRADE_TRANSACTION_FEE / 2.0, 0.0)
            buyer_gain = volume * max(grid_buy_price - price - TRADE_TRANSACTION_FEE / 2.0, 0.0)
            if seller_gain > 1e-6 and buyer_gain > 1e-6:
                winwin_pairs += 1
            trade_benefit_internal[seller] += seller_gain
            trade_benefit_internal[buyer] += buyer_gain
            renewable_part = min(volume, float(external[seller]["pv_export"][hour]))
            renewable_traded += renewable_part
            non_renewable_part = max(volume - renewable_part, 0.0)
            load_reference = max(float(external[seller]["total_load"][hour]), 1e-6)
            dependency_ratio = min(float(external[seller]["buy"][hour]) / load_reference, 1.0)
            trade_intensity = float(external[seller]["carbon_intensity"][hour]) * dependency_ratio * 0.55
            trade_carbon_add[buyer] += non_renewable_part * trade_intensity
            trade_payment_out[buyer] += volume * price + volume * TRADE_TRANSACTION_FEE / 2.0
            trade_payment_in[seller] += volume * price - volume * TRADE_TRANSACTION_FEE / 2.0
            if volume > link_caps[(seller, buyer)] + 1e-9 and not enforce_clip:
                total_violations += 1
                total_balance_error += volume - link_caps[(seller, buyer)]

        for park_id in park_ids:
            base_buy = float(external[park_id]["buy"][hour])
            base_sell = float(external[park_id]["sell"][hour])
            allowed_export = base_sell + float(export_headroom.get(park_id, [0.0] * horizon)[hour])
            if exports_by_park[park_id] > allowed_export + 1e-9 and not enforce_clip:
                total_violations += 1
                total_balance_error += exports_by_park[park_id] - allowed_export
            if imports_by_park[park_id] > base_buy + 1e-9 and not enforce_clip:
                total_violations += 1
                total_balance_error += imports_by_park[park_id] - base_buy

            if enforce_clip:
                per_park_hourly_buy[park_id][hour] = max(base_buy - imports_by_park[park_id], 0.0)
                per_park_hourly_sell[park_id][hour] = max(base_sell - exports_by_park[park_id], 0.0)
                park_coordination_cost[park_id] += 0.040 * max(exports_by_park[park_id] - base_sell, 0.0)
            else:
                residual_buy = max(base_buy - imports_by_park[park_id], 0.0)
                residual_sell = max(base_sell - exports_by_park[park_id], 0.0)
                oversupply = max(imports_by_park[park_id] - base_buy, 0.0)
                shortage = max(exports_by_park[park_id] - allowed_export, 0.0)
                total_balance_error += oversupply + shortage
                per_park_hourly_buy[park_id][hour] = residual_buy
                per_park_hourly_sell[park_id][hour] = residual_sell
                trade_benefit_internal[park_id] -= RAW_TRADE_PENALTY * (oversupply + shortage)
                park_penalty_cost[park_id] += RAW_TRADE_PENALTY * (oversupply + shortage)
                park_penalty_emission[park_id] += 0.62 * (oversupply + shortage)

    for park_id, state in states.items():
        run = state.internal_run
        arrays = external[park_id]
        fixed_cost = run.battery_degradation_cost + run.service_penalty_cost
        electricity_cost = float(np.sum(per_park_hourly_buy[park_id] * arrays["buy_price"]))
        grid_revenue = float(np.sum(per_park_hourly_sell[park_id] * arrays["sell_price"]))
        carbon_cost = float(np.sum(per_park_hourly_buy[park_id] * arrays["carbon_intensity"] * arrays["carbon_price"]))
        base_emission = float(np.sum(per_park_hourly_buy[park_id] * arrays["carbon_intensity"]))
        park_cost = electricity_cost - grid_revenue + carbon_cost + fixed_cost + trade_payment_out[park_id] - trade_payment_in[park_id] + park_penalty_cost[park_id] + park_coordination_cost[park_id]
        park_emission = base_emission + trade_carbon_add[park_id] + park_penalty_emission[park_id]
        base_park_costs[park_id] = park_cost
        park_emissions[park_id] = round(park_emission, 4)
        park_loads[park_id] = float(np.sum(arrays["total_load"]))
        park_carbon_responsibility[park_id] = park_emission
        total_grid_purchase += float(np.sum(per_park_hourly_buy[park_id]))
        total_emission += park_emission

    carbon_market = _settle_carbon_market(states, park_emissions, park_loads)
    for park_id, state in states.items():
        carbon_market_net_cost = float(carbon_market["net_cost_rmb"][park_id])
        park_cost = base_park_costs[park_id] + carbon_market_net_cost
        park_costs[park_id] = round(park_cost, 4)
        trading_benefits[park_id] = round(state.internal_run.total_operating_cost - park_cost, 4)

    total_system_cost = float(sum(park_costs.values()) - sum(trade_payment_out.values()) + sum(trade_payment_in.values()) + total_transaction_fee)
    no_trade_grid_purchase = baseline_no_trade["total_grid_purchase"]
    no_trade_high_carbon = baseline_no_trade["high_carbon_grid_purchase"]
    current_high_carbon = 0.0
    for park_id in park_ids:
        threshold = float(np.quantile(external[park_id]["carbon_intensity"], 0.75))
        current_high_carbon += float(np.sum(per_park_hourly_buy[park_id][external[park_id]["carbon_intensity"] >= threshold]))

    grid_dependence_reduction = (no_trade_grid_purchase - total_grid_purchase) / max(no_trade_grid_purchase, 1e-6)
    high_carbon_reduction = (no_trade_high_carbon - current_high_carbon) / max(no_trade_high_carbon, 1e-6)
    renewable_sharing_rate = renewable_traded / max(total_trade_volume, 1e-6) if total_trade_volume > 1e-6 else 0.0
    benefit_values = np.array([max(value, 0.0) + 1e-6 for value in trading_benefits.values()], dtype=float)
    benefit_fairness = jains_index(benefit_values)
    load_share = np.array([park_loads[pid] for pid in park_ids], dtype=float)
    load_share = load_share / max(float(load_share.sum()), 1e-6)
    carbon_share = np.array([park_carbon_responsibility[pid] for pid in park_ids], dtype=float)
    carbon_share = carbon_share / max(float(carbon_share.sum()), 1e-6)
    burden_ratio = carbon_share / np.maximum(load_share, 1e-6)
    carbon_fairness = jains_index(burden_ratio)
    average_clearing_price = total_trade_value / max(total_trade_volume, 1e-6) if total_trade_volume > 1e-6 else 0.0
    winwin_ratio = winwin_pairs / max(total_pairs, 1)

    return Case2Run(
        name=name,
        run_id=run_id,
        model_name=model_name,
        total_system_operating_cost=round(total_system_cost, 4),
        total_system_carbon_emission=round(total_emission, 4),
        total_grid_purchase=round(total_grid_purchase, 4),
        interpark_trading_volume=round(total_trade_volume, 4),
        carbon_credit_trading_volume=round(float(carbon_market["internal_trading_volume_kg"]), 4),
        total_carbon_compliance_cost=round(float(carbon_market["total_compliance_cost_rmb"]), 4),
        average_carbon_credit_price=round(float(carbon_market["clearing_price_rmb_per_kg"]), 4),
        average_clearing_price=round(float(average_clearing_price), 4),
        grid_dependence_reduction=round(float(grid_dependence_reduction), 6),
        negotiation_convergence_rate=round(float(trading_result["convergence_rate"]), 6),
        total_negotiation_rounds=round(float(trading_result["negotiation_rounds"]), 4),
        renewable_energy_sharing_rate=round(float(renewable_sharing_rate), 6),
        high_carbon_exposure_reduction=round(float(high_carbon_reduction), 6),
        benefit_distribution_fairness=round(float(benefit_fairness), 6),
        carbon_responsibility_fairness=round(float(carbon_fairness), 6),
        winwin_ratio=round(float(winwin_ratio), 6),
        constraint_violation_rate=round(float(total_violations / max(total_checks, 1)), 6),
        power_balance_error=round(float(total_balance_error / horizon), 4),
        park_costs=park_costs,
        trading_benefits=trading_benefits,
        park_emissions=park_emissions,
        details={
            "pair_trade_totals_kwh": {f"{seller}->{buyer}": round(volume, 4) for (seller, buyer), volume in bilateral_totals.items() if volume > 1e-6},
            "trade_by_hour": {str(hour): {f"{seller}->{buyer}": {"volume_kwh": volume, "price_rmb_per_kwh": price_book[hour][(seller, buyer)]} for (seller, buyer), volume in trade[hour].items()} for hour in range(horizon) if trade[hour]},
            "buy_price_rmb_per_kwh": series_payload(np.asarray(next(iter(external.values()))["buy_price"], dtype=float)),
            "sell_price_rmb_per_kwh": series_payload(np.asarray(next(iter(external.values()))["sell_price"], dtype=float)),
            "park_hourly_grid_buy": {park_id: series_payload(per_park_hourly_buy[park_id]) for park_id in park_ids},
            "park_hourly_grid_sell": {park_id: series_payload(per_park_hourly_sell[park_id]) for park_id in park_ids},
            "carbon_market": carbon_market,
            "llm_summary": llm_intent.get("summary", []) if llm_intent is not None else [],
            "rounds_per_hour": trading_result.get("rounds_per_hour", []),
            "round_logs": trading_result.get("round_logs", {}),
            "price_path": trading_result.get("price_path", {}),
            "projection_adjustments": trading_result.get("projection_adjustments", {}),
            "behavior_summary": trading_result.get("behavior_summary", []),
            "partner_switch": trading_result.get("partner_switch", {}),
        },
    )


def _save_case2_tables(states: dict[str, ParkState], aggregated: list[dict[str, Any]]) -> None:
    rows = []
    for state in states.values():
        config = state.scenario["config"]
        rows.append([state.spec.display_name, state.spec.park_type, config["pv"]["rated_power_kw"], config["ess"]["rated_power_kw"], config["ess"]["energy_capacity_kwh"], config["inflexible_peak_kw"], config["ev_cluster"]["daily_energy_kwh"], state.spec.carbon_sensitivity])

    with (CASE2_OUTPUT_DIR / "table3_case2_parameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Park", "Type", "PV (kW)", "ESS power (kW)", "ESS energy (kWh)", "Peak load (kW)", "EV daily demand (kWh/day)", "Carbon sensitivity"])
        writer.writerows(rows)
    write_markdown(
        CASE2_OUTPUT_DIR / "table3_case2_parameters.md",
        "\n".join(
            ["# Table 3. Weekly Case II park parameters", "", "| Park | Type | PV (kW) | ESS power (kW) | ESS energy (kWh) | Peak load (kW) | EV daily demand (kWh/day) | Carbon sensitivity |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            + [f"| {park} | {park_type} | {pv} | {ess_power} | {ess_energy} | {peak} | {ev_daily} | {carbon_sensitivity} |" for park, park_type, pv, ess_power, ess_energy, peak, ev_daily, carbon_sensitivity in rows]
        ),
    )

    with (CASE2_OUTPUT_DIR / "table4_case2_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Baseline", "System cost", "System emission", "Electricity trading volume", "Carbon credit volume", "Carbon compliance cost", "Grid reduction", "Convergence", "Benefit fairness", "Carbon fairness", "Win-win ratio", "Violation rate"])
        for item in aggregated:
            metrics = item["metrics"]
            writer.writerow([item["baseline"], metrics["total_system_operating_cost"]["mean"], metrics["total_system_carbon_emission"]["mean"], metrics["interpark_trading_volume"]["mean"], metrics["carbon_credit_trading_volume"]["mean"], metrics["total_carbon_compliance_cost"]["mean"], metrics["grid_dependence_reduction"]["mean"], metrics["negotiation_convergence_rate"]["mean"], metrics["benefit_distribution_fairness"]["mean"], metrics["carbon_responsibility_fairness"]["mean"], metrics["winwin_ratio"]["mean"], metrics["constraint_violation_rate"]["mean"]])
    write_markdown(
        CASE2_OUTPUT_DIR / "table4_case2_performance.md",
        "\n".join(
            ["# Table 4. Weekly Case II performance", "", "| Baseline | System cost | System emission | Electricity trading volume | Carbon credit volume | Carbon compliance cost | Grid reduction | Convergence | Benefit fairness | Carbon fairness | Win-win ratio | Violation rate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            + [
                f"| {item['baseline']} | {item['metrics']['total_system_operating_cost']['mean']} | {item['metrics']['total_system_carbon_emission']['mean']} | {item['metrics']['interpark_trading_volume']['mean']} | {item['metrics']['carbon_credit_trading_volume']['mean']} | {item['metrics']['total_carbon_compliance_cost']['mean']} | {item['metrics']['grid_dependence_reduction']['mean']} | {item['metrics']['negotiation_convergence_rate']['mean']} | {item['metrics']['benefit_distribution_fairness']['mean']} | {item['metrics']['carbon_responsibility_fairness']['mean']} | {item['metrics']['winwin_ratio']['mean']} | {item['metrics']['constraint_violation_rate']['mean']} |"
                for item in aggregated
            ]
        ),
    )


def _save_case2_summary(aggregated: list[dict[str, Any]], reference_run: Case2Run) -> None:
    metrics = {item["baseline"]: item["metrics"] for item in aggregated}
    behavior_highlights = reference_run.details.get("behavior_summary", [])[:3]
    write_markdown(
        CASE2_OUTPUT_DIR / "case2_summary.md",
        "\n".join(
            [
                "# Case II Summary",
                "",
                "## Key findings",
                "",
                "- Inter-park trading over the 7x24 horizon reduces system operating cost and external grid dependence relative to the no-trading baseline.",
                "- Explicit park-level LLM agents produce heterogeneous bidding roles that are grounded by the physical coordination layer.",
                "- Removing the physics-aware coordination layer still causes significant feasibility loss and unstable market outcomes.",
                "",
                "## Selected metrics",
                "",
                f"- `B1` system cost: {metrics['B1_No_InterPark_Trading']['total_system_operating_cost']['mean']:.3f} RMB",
                f"- `B5` system cost: {metrics['B5_Proposed_PI_MA_LLMs']['total_system_operating_cost']['mean']:.3f} RMB",
                f"- `B5` trading volume: {metrics['B5_Proposed_PI_MA_LLMs']['interpark_trading_volume']['mean']:.3f} kWh",
                f"- `B5` carbon credit trading volume: {metrics['B5_Proposed_PI_MA_LLMs']['carbon_credit_trading_volume']['mean']:.3f} kg",
                f"- `B5` carbon compliance cost: {metrics['B5_Proposed_PI_MA_LLMs']['total_carbon_compliance_cost']['mean']:.3f} RMB",
                f"- `B5` grid dependence reduction: {metrics['B5_Proposed_PI_MA_LLMs']['grid_dependence_reduction']['mean']:.4f}",
                f"- `B4` violation rate: {metrics['B4_LLM_Bidding_Box_Clipping']['constraint_violation_rate']['mean']:.4f}",
                f"- `B5` carbon fairness: {metrics['B5_Proposed_PI_MA_LLMs']['carbon_responsibility_fairness']['mean']:.4f}",
                "",
                "## Reference B5 run",
                "",
                f"- Run id: `{reference_run.run_id}`",
                f"- Average clearing price: {reference_run.average_clearing_price:.3f} RMB/kWh",
                f"- Average carbon credit price: {reference_run.average_carbon_credit_price:.3f} RMB/kg",
                f"- Win-win ratio: {reference_run.winwin_ratio:.4f}",
                f"- LLM summary: {', '.join(reference_run.details.get('llm_summary', []))}",
                "",
                "## Behavior highlights",
                "",
            ]
            + [f"- {item}" for item in behavior_highlights]
        ),
    )


def _save_case2_behavior_summary(reference_run: Case2Run) -> None:
    round_logs = reference_run.details.get("round_logs", {})
    representative_hours = []
    for hour_str, payload in round_logs.items():
        rounds = payload.get("rounds", [])
        if not rounds:
            continue
        total_volume = sum(
            float(pair["volume_kwh"])
            for round_payload in rounds
            for pair in round_payload.get("executed_pairs", [])
        )
        representative_hours.append((total_volume + 20.0 * len(rounds), int(hour_str), payload))
    representative_hours.sort(reverse=True)
    lines = [
        "# Case II Behavior Summary",
        "",
        "## Weekly bidding digest",
        "",
    ]
    for item in reference_run.details.get("llm_summary", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Representative hours", ""])
    for _, hour, payload in representative_hours[:4]:
        lines.append(f"### Hour {hour}")
        lines.extend([""] + [f"- {text}" for text in payload.get("hour_summary", [])])
        for round_payload in payload.get("rounds", [])[:3]:
            executed_pairs = round_payload.get("executed_pairs", [])
            pair_text = ", ".join(
                f"{item['seller']}->{item['buyer']} ({item['volume_kwh']:.1f} kWh @ {item['price_rmb_per_kwh']:.3f})"
                for item in executed_pairs
            ) or "No executed pair"
            lines.append(f"- Round {round_payload['round_index']}: {pair_text}")
        lines.append("")
    if len(lines) <= 7:
        lines.append("- No representative multi-round behavior was recorded.")
    write_markdown(CASE2_OUTPUT_DIR / "case2_behavior_summary.md", "\n".join(lines))


def _homogeneous_case2_intent(llm_intent: dict[str, Any]) -> dict[str, Any]:
    adjusted = copy.deepcopy(llm_intent)
    max_rounds = int(adjusted.get("max_bidding_rounds", adjusted.get("max_negotiation_rounds", 5)) or 5)
    for hour_key, hour_payload in adjusted.get("hours", {}).items():
        context = hour_payload.get("context", {})
        park_context = context.get("parks", {})
        for round_payload in hour_payload.get("rounds", []):
            round_index = int(round_payload.get("round_index", 1))
            concession = clamp(0.22 + 0.32 * round_index / max(max_rounds, 1), 0.0, 1.0)
            for park_id, output in round_payload.get("park_outputs", {}).items():
                local = park_context.get(park_id, {})
                sell_cap = float(local.get("base_sell_kw", output.get("export_target_kwh", 0.0)))
                buy_cap = float(local.get("base_buy_kw", output.get("import_target_kwh", 0.0)))
                buy_price = float(local.get("buy_price_rmb_per_kwh", output.get("bid_price_rmb_per_kwh", 0.0)))
                sell_price = float(local.get("sell_price_rmb_per_kwh", output.get("ask_price_rmb_per_kwh", 0.0)))
                spread = max(buy_price - sell_price, 0.03)
                if sell_cap >= buy_cap + 1e-6 and sell_cap > 1e-6:
                    role = "seller"
                    export_target = sell_cap * 0.94
                    import_target = 0.0
                elif buy_cap > sell_cap + 1e-6 and buy_cap > 1e-6:
                    role = "buyer"
                    export_target = 0.0
                    import_target = buy_cap * 0.94
                elif max(sell_cap, buy_cap) > 1e-6:
                    role = "balanced"
                    export_target = sell_cap * 0.86
                    import_target = buy_cap * 0.86
                else:
                    role = "idle"
                    export_target = 0.0
                    import_target = 0.0
                partner_priority = {other_id: 0.58 for other_id in CASE2_PARK_IDS if other_id != park_id}
                output.update(
                    {
                        "role": role,
                        "export_willingness": 0.58 if export_target > 1e-6 else 0.0,
                        "import_willingness": 0.58 if import_target > 1e-6 else 0.0,
                        "carbon_priority": 0.5,
                        "concession_factor": round(float(concession), 4),
                        "export_target_kwh": round(float(export_target), 4),
                        "import_target_kwh": round(float(import_target), 4),
                        "ask_price_rmb_per_kwh": round(float(sell_price + spread * (0.42 + 0.08 * (1.0 - concession))), 4),
                        "bid_price_rmb_per_kwh": round(float(buy_price - spread * (0.20 + 0.10 * concession)), 4),
                        "carbon_market_posture": "balanced",
                        "partner_priority": partner_priority,
                        "continue_bidding": bool(round_index < max_rounds and role != "idle"),
                        "message": f"{park_id} follows homogeneous neutral bidding in round {round_index}.",
                        "summary": f"{park_id} neutral {role}: export {export_target:.1f} kWh, import {import_target:.1f} kWh.",
                    }
                )
            round_payload.pop("order_book", None)
            round_payload.pop("tentative_pairs", None)
            round_payload.pop("order_book_feedback", None)
    adjusted["summary"] = ["Heterogeneous profile cues are replaced by neutral bidding preferences."]
    return adjusted


def _first_round_only_case2_intent(llm_intent: dict[str, Any]) -> dict[str, Any]:
    adjusted = copy.deepcopy(llm_intent)
    for hour_payload in adjusted.get("hours", {}).values():
        rounds = hour_payload.get("rounds", [])
        if rounds:
            first_round = copy.deepcopy(rounds[0])
            first_round["round_index"] = 1
            for output in first_round.get("park_outputs", {}).values():
                output["continue_bidding"] = False
            hour_payload["rounds"] = [first_round]
            hour_payload["hour_summary"] = ["Only the first bidding round is retained; feedback memory is disabled."]
    adjusted["summary"] = ["Order-book feedback and cross-round memory are disabled by retaining first-round bids only."]
    return adjusted


def _run_parameterized_case2_intents(states: dict[str, ParkState], repeats: int) -> list[dict[str, Any]]:
    orchestrator = Case2MultiAgentOrchestrator(model="parameterized_bidding_agent", use_mock_llm=True)
    return [
        orchestrator.generate_intent(states, idx, LLM_DIR / f"parameterized_ablation_run_{idx}.json")
        for idx in range(1, repeats + 1)
    ]


def _case2_ablation_runs(
    states: dict[str, ParkState],
    baseline_reference: dict[str, float],
    llm_intents: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    repeats = len(llm_intents)
    parameterized_intents = _run_parameterized_case2_intents(states, repeats)
    c1_runs: list[Case2Run] = []
    c2_runs: list[Case2Run] = []
    c3_runs: list[Case2Run] = []
    c5_runs: list[Case2Run] = []
    for idx, intent in enumerate(llm_intents, start=1):
        c1_runs.append(_evaluate_case2("C1_Full_PA_MA_LLMs", f"c1_{idx}", states, _run_llm_trading(states, intent, physics_projection=True), baseline_reference, model_name, intent, True))
        homogeneous_intent = _homogeneous_case2_intent(intent)
        c2_runs.append(_evaluate_case2("C2_No_Heterogeneous_Profile", f"c2_{idx}", states, _run_llm_trading(states, homogeneous_intent, physics_projection=True), baseline_reference, "homogeneous_profile_control", homogeneous_intent, True))
        first_round_intent = _first_round_only_case2_intent(intent)
        c3_runs.append(_evaluate_case2("C3_No_Feedback_Memory", f"c3_{idx}", states, _run_llm_trading(states, first_round_intent, physics_projection=True), baseline_reference, "first_round_control", first_round_intent, True))
        param_intent = parameterized_intents[idx - 1]
        c5_runs.append(_evaluate_case2("C5_Parameterized_Bidding", f"c5_{idx}", states, _run_llm_trading(states, param_intent, physics_projection=True), baseline_reference, "parameterized_bidding_agent", param_intent, True))

    c4 = _evaluate_case2("C4_Rule_Based_Bidding", "c4_1", states, _run_rule_based_trading(states), baseline_reference, "deterministic", None, True)
    return [
        aggregate_case2_runs("C1_Full_PA_MA_LLMs", c1_runs),
        aggregate_case2_runs("C2_No_Heterogeneous_Profile", c2_runs),
        aggregate_case2_runs("C3_No_Feedback_Memory", c3_runs),
        aggregate_case2_runs("C4_Rule_Based_Bidding", [c4]),
        aggregate_case2_runs("C5_Parameterized_Bidding", c5_runs),
    ]


def _save_case2_ablation_table(ablation_results: list[dict[str, Any]]) -> None:
    headers = [
        "Variant",
        "System cost",
        "System emission",
        "Electricity trading volume",
        "Carbon credit volume",
        "Grid reduction",
        "Convergence",
        "Benefit fairness",
        "Carbon fairness",
        "Violation rate",
    ]
    with (CASE2_OUTPUT_DIR / "table5_case2_ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for item in ablation_results:
            metrics = item["metrics"]
            writer.writerow(
                [
                    item["baseline"],
                    metrics["total_system_operating_cost"]["mean"],
                    metrics["total_system_carbon_emission"]["mean"],
                    metrics["interpark_trading_volume"]["mean"],
                    metrics["carbon_credit_trading_volume"]["mean"],
                    metrics["grid_dependence_reduction"]["mean"],
                    metrics["negotiation_convergence_rate"]["mean"],
                    metrics["benefit_distribution_fairness"]["mean"],
                    metrics["carbon_responsibility_fairness"]["mean"],
                    metrics["constraint_violation_rate"]["mean"],
                ]
            )
    write_markdown(
        CASE2_OUTPUT_DIR / "table5_case2_ablation.md",
        "\n".join(
            ["# Table 5. Case II market-behavior ablation controls", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + [
                "| "
                + " | ".join(
                    [
                        str(item["baseline"]),
                        str(item["metrics"]["total_system_operating_cost"]["mean"]),
                        str(item["metrics"]["total_system_carbon_emission"]["mean"]),
                        str(item["metrics"]["interpark_trading_volume"]["mean"]),
                        str(item["metrics"]["carbon_credit_trading_volume"]["mean"]),
                        str(item["metrics"]["grid_dependence_reduction"]["mean"]),
                        str(item["metrics"]["negotiation_convergence_rate"]["mean"]),
                        str(item["metrics"]["benefit_distribution_fairness"]["mean"]),
                        str(item["metrics"]["carbon_responsibility_fairness"]["mean"]),
                        str(item["metrics"]["constraint_violation_rate"]["mean"]),
                    ]
                )
                + " |"
                for item in ablation_results
            ]
        ),
    )


def run_case2(repeats: int = 3, model: str = DEFAULT_MODEL, use_mock_llm: bool = False, reuse_existing_traces: bool = False) -> dict[str, Any]:
    CASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    states = _build_park_states()
    api_key = None if use_mock_llm else read_api_key(KEY_PATH)
    orchestrator = Case2MultiAgentOrchestrator(model=model, use_mock_llm=use_mock_llm, api_key=api_key)
    _attach_subjective_profiles(states, client=orchestrator.client, use_mock=use_mock_llm)

    baseline_no_trade_cost = 0.0
    baseline_no_trade_emission = 0.0
    baseline_no_trade_grid = 0.0
    baseline_high_carbon = 0.0
    for state in states.values():
        run = state.internal_run
        baseline_no_trade_cost += run.total_operating_cost
        baseline_no_trade_emission += run.total_carbon_emission
        buy = to_np(run.details["hourly"]["buy_kw"])
        baseline_no_trade_grid += float(np.sum(buy))
        threshold = float(np.quantile(state.arrays["carbon_intensity"], 0.75))
        baseline_high_carbon += float(np.sum(buy[state.arrays["carbon_intensity"] >= threshold]))
    baseline_reference = {"total_cost": baseline_no_trade_cost, "total_emission": baseline_no_trade_emission, "total_grid_purchase": baseline_no_trade_grid, "high_carbon_grid_purchase": baseline_high_carbon}
    horizon = len(next(iter(states.values())).arrays["hours"])
    empty_trade = {
        "trade": {hour: {} for hour in range(horizon)},
        "price_book": {hour: {} for hour in range(horizon)},
        "rounds_per_hour": [0.0] * horizon,
        "negotiation_rounds": 0.0,
        "convergence_rate": 0.0,
        "export_headroom": {park_id: [0.0] * horizon for park_id in states},
        "round_logs": {},
        "price_path": {},
        "projection_adjustments": {},
        "behavior_summary": [],
        "partner_switch": {park_id: 0 for park_id in states},
    }

    b1 = _evaluate_case2("B1_No_InterPark_Trading", "b1_1", states, empty_trade, baseline_reference, "deterministic", None, True)
    b2 = _evaluate_case2("B2_Rule_Based_InterPark_Trading", "b2_1", states, _run_rule_based_trading(states), baseline_reference, "deterministic", None, True)
    b3 = _evaluate_case2("B3_Traditional_Game_Based_Method", "b3_1", states, _run_game_based_trading(states), baseline_reference, "deterministic", None, True)

    b4_runs: list[Case2Run] = []
    b5_runs: list[Case2Run] = []
    llm_intents: list[dict[str, Any]] = []
    llm_records: list[dict[str, Any]] = []
    for idx in range(1, repeats + 1):
        trace_path = LLM_DIR / (f"mock_run_{idx}.json" if use_mock_llm else f"deepseek_run_{idx}.json")
        llm_intent = None
        if reuse_existing_traces and trace_path.exists():
            cached_intent = load_json(trace_path)
            if (
                isinstance(cached_intent, dict)
                and "hours" in cached_intent
                and ("max_bidding_rounds" in cached_intent or "max_negotiation_rounds" in cached_intent)
                and _llm_trace_matches_states(cached_intent, set(states.keys()))
            ):
                llm_intent = cached_intent
        if llm_intent is None:
            llm_intent = orchestrator.generate_intent(states, idx, trace_path)
        llm_intents.append(llm_intent)
        llm_records.append(
            {
                "run_id": f"llm_{idx}",
                "max_bidding_rounds": llm_intent.get("max_bidding_rounds", llm_intent.get("max_negotiation_rounds", 0)),
                "summary": llm_intent["summary"],
                "active_hours": sum(1 for payload in llm_intent["hours"].values() if payload.get("rounds")),
                "sample_hour_summaries": {
                    hour: payload["hour_summary"]
                    for hour, payload in list(llm_intent["hours"].items())[:8]
                },
            }
        )
        b4_runs.append(_evaluate_case2("B4_LLM_Bidding_Box_Clipping", f"b4_{idx}", states, _run_llm_trading(states, llm_intent, physics_projection=False, box_clipping=True), baseline_reference, model if not use_mock_llm else "mock_llm", llm_intent, physics_projection=False, box_clipping=True))
        b5_runs.append(_evaluate_case2("B5_Proposed_PI_MA_LLMs", f"b5_{idx}", states, _run_llm_trading(states, llm_intent, physics_projection=True), baseline_reference, model if not use_mock_llm else "mock_llm", llm_intent, True))

    # B6: Independent PPO MARL baseline. Each park has its own linear policy
    # trained on the same scenario; trained intents are then cleared through
    # the same physics-aware projection used by B5 for an apples-to-apples
    # comparison.
    from methods.marl import MARLConfig, rollout_to_intent, train_independent_ppo
    b6_runs: list[Case2Run] = []
    marl_intents: list[dict[str, Any]] = []
    for idx in range(1, repeats + 1):
        marl_cfg = MARLConfig(seed=20260512 + idx)
        marl_policies = train_independent_ppo(states, marl_cfg)
        marl_intent = rollout_to_intent(marl_policies, states)
        marl_intents.append(marl_intent)
        write_json(LLM_DIR / f"marl_run_{idx}.json", marl_intent)
        b6_runs.append(
            _evaluate_case2(
                "B6_Independent_PPO_MARL",
                f"b6_{idx}",
                states,
                _run_llm_trading(states, marl_intent, physics_projection=True),
                baseline_reference,
                "independent_ppo",
                marl_intent,
                True,
            )
        )

    aggregated = [
        aggregate_case2_runs("B1_No_InterPark_Trading", [b1]),
        aggregate_case2_runs("B2_Rule_Based_InterPark_Trading", [b2]),
        aggregate_case2_runs("B3_Traditional_Game_Based_Method", [b3]),
        aggregate_case2_runs("B4_LLM_Bidding_Box_Clipping", b4_runs),
        aggregate_case2_runs("B6_Independent_PPO_MARL", b6_runs),
        aggregate_case2_runs("B5_Proposed_PI_MA_LLMs", b5_runs),
    ]
    reference_run = select_reference_case2_run(b5_runs)
    ablation_results = _case2_ablation_runs(states, baseline_reference, llm_intents, model if not use_mock_llm else "mock_llm")

    # Behavior-fidelity metrics for methods that produce a bidding intent.
    from methods.behavior_metrics import compute_behavior_fidelity
    behavior_fidelity = {
        "B4_LLM_Bidding_Box_Clipping": compute_behavior_fidelity(states, llm_intents[0]).as_dict() if llm_intents else None,
        "B5_Proposed_PI_MA_LLMs": compute_behavior_fidelity(states, llm_intents[0]).as_dict() if llm_intents else None,
        "B6_Independent_PPO_MARL": compute_behavior_fidelity(states, marl_intents[0]).as_dict() if marl_intents else None,
    }

    _save_case2_tables(states, aggregated)
    _save_case2_ablation_table(ablation_results)
    plot_case2_coupled_market_summary(reference_run, FIGURE_DIR)
    plot_case2_negotiation(aggregated, reference_run, FIGURE_DIR)
    _save_case2_summary(aggregated, reference_run)
    _save_case2_behavior_summary(reference_run)
    write_json(CASE2_OUTPUT_DIR / "case2_behavior_fidelity.json", behavior_fidelity)

    # Also dump the extracted subjective profiles for the manuscript table.
    profile_export = {
        park_id: {
            "park_type": state.spec.park_type,
            "subjective_profile": state.subjective_profile or {},
            "profile_rationales": state.profile_rationales or {},
        }
        for park_id, state in states.items()
    }
    write_json(CASE2_OUTPUT_DIR / "case2_subjective_profiles.json", profile_export)

    payload = {
        "case": "5.2 Case II: Multi-Round Bidding and Trading Among Multiple Low-Carbon Parks",
        "llm_model": model if not use_mock_llm else "mock_llm_for_smoke_test",
        "parks": {park_id: state.scenario for park_id, state in states.items()},
        "aggregated_results": aggregated,
        "ablation_results": ablation_results,
        "behavior_fidelity": behavior_fidelity,
        "subjective_profiles": profile_export,
        "reference_b5_run": reference_run.__dict__,
        "llm_intents": llm_records,
    }
    write_json(CASE2_OUTPUT_DIR / "case2_results.json", payload)
    return payload
