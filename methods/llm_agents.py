from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from methods.common import CASE2_PARK_IDS, base_arrays, carbon_quota_factor
from utils.constants import DEEPSEEK_API_URL, DEFAULT_MODEL
from utils.io_utils import write_json
from utils.math_utils import clamp, series_payload


LLM_GENERATION_TEMPERATURE = 0.2
LLM_REPAIR_TEMPERATURE = 0.0
MAX_LLM_ATTEMPTS = 3


class DeepSeekJSONClient:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    @staticmethod
    def _strip_code_fence(raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        return text.strip()

    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict[str, Any]:
        text = DeepSeekJSONClient._strip_code_fence(raw_text)
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model response.")
        candidate = text[start:]
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        end = text.rfind("}")
        if end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        balanced = candidate
        bracket_gap = balanced.count("[") - balanced.count("]")
        brace_gap = balanced.count("{") - balanced.count("}")
        if bracket_gap > 0:
            balanced += "]" * bracket_gap
        if brace_gap > 0:
            balanced += "}" * brace_gap
        parsed = json.loads(balanced)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a JSON object.")
        return parsed

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            DEEPSEEK_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _repair_json_text(self, raw_text: str) -> dict[str, Any]:
        repair_payload = {
            "model": self.model,
            "temperature": LLM_REPAIR_TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": "You repair malformed JSON emitted by another model. Return one valid JSON object only. Preserve numeric values and fields when possible, fill missing optional strings with concise placeholders, and do not add markdown fences.",
                },
                {
                    "role": "user",
                    "content": raw_text,
                },
            ],
        }
        repaired_response = self._request_json(repair_payload)
        repaired_text = repaired_response["choices"][0]["message"]["content"].strip()
        parsed = self._extract_json_payload(repaired_text)
        parsed["_repair_response"] = repaired_response
        return parsed

    def generate_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": LLM_GENERATION_TEMPERATURE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        response_payload: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                response_payload = self._request_json(payload)
                raw_text = response_payload["choices"][0]["message"]["content"].strip()
                try:
                    parsed = self._extract_json_payload(raw_text)
                except Exception:
                    parsed = self._repair_json_text(raw_text)
                parsed["_raw_response"] = response_payload
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                if attempt == MAX_LLM_ATTEMPTS - 1:
                    raise RuntimeError(f"DeepSeek API request failed: {exc.code} {body}") from exc
                time.sleep(2.0 + attempt)
            except Exception as exc:
                if attempt == MAX_LLM_ATTEMPTS - 1:
                    raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc
                time.sleep(2.0 + attempt)

        if response_payload is None or parsed is None:
            raise RuntimeError("DeepSeek API did not return a response payload.")
        return parsed


def _ensure_series(values: Any, horizon: int, low: float, high: float, fallback: np.ndarray) -> np.ndarray:
    if not isinstance(values, list) or len(values) != horizon:
        return fallback.copy()
    return np.asarray([clamp(float(value), low, high) for value in values], dtype=float)


def _case1_mock_operator(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    price_norm = arrays["buy_price"] / max(float(arrays["buy_price"].max()), 1.0)
    carbon_norm = arrays["carbon_intensity"] / max(float(arrays["carbon_intensity"].max()), 1.0)
    pv_norm = arrays["pv"] / max(float(arrays["pv"].max()), 1.0)
    econ_focus = np.clip(0.52 + 0.38 * price_norm - 0.10 * pv_norm, 0.0, 1.0)
    carbon_focus = np.clip(0.44 + 0.40 * carbon_norm - 0.18 * pv_norm, 0.0, 1.0)
    return {
        "econ_focus": series_payload(econ_focus, digits=4),
        "carbon_focus": series_payload(carbon_focus, digits=4),
        "coordination_message": "Align weekly flexible demand with PV-rich windows and suppress evening carbon peaks.",
        "summary": [
            "The operator emphasizes weekly low-carbon coordination over the 7x24 horizon.",
            "Midday PV windows should absorb flexible load and charging demand.",
            "Evening peaks should be softened with ESS discharge and tighter EV prioritization.",
        ],
        "model_name": DEFAULT_MODEL,
    }


def _case1_mock_resource(name: str, arrays: dict[str, np.ndarray], operator: dict[str, Any]) -> dict[str, Any]:
    price_norm = arrays["buy_price"] / max(float(arrays["buy_price"].max()), 1.0)
    carbon_norm = arrays["carbon_intensity"] / max(float(arrays["carbon_intensity"].max()), 1.0)
    pv_norm = arrays["pv"] / max(float(arrays["pv"].max()), 1.0)
    econ_focus = np.asarray(operator["econ_focus"], dtype=float)
    carbon_focus = np.asarray(operator["carbon_focus"], dtype=float)

    if name == "ess_agent":
        signal = np.clip(0.65 * pv_norm - 0.55 * price_norm - 0.35 * carbon_norm, -1.0, 1.0)
        summary = "ESS charges in PV-rich windows and discharges during high-price high-carbon peaks."
    elif name == "hvac_agent":
        temp_norm = arrays["temperature"] / max(float(arrays["temperature"].max()), 1.0)
        signal = np.clip(0.62 * pv_norm + 0.28 * temp_norm - 0.34 * carbon_focus, -1.0, 1.0)
        summary = "HVAC shifts thermal consumption toward daytime comfort windows with stronger PV support."
    elif name == "service_load_agent":
        business_hours = ((arrays["hour_of_day"] >= 8) & (arrays["hour_of_day"] <= 18)).astype(float)
        signal = np.clip(0.72 * pv_norm + 0.18 * business_hours - 0.30 * carbon_focus, -1.0, 1.0)
        summary = "Shiftable service demand is pulled toward productive daylight periods and away from evening peaks."
    elif name == "ev_agent":
        in_window = ((arrays["hour_of_day"] >= 7) & (arrays["hour_of_day"] <= 20)).astype(float)
        signal = np.clip(in_window * (0.25 + 0.62 * pv_norm - 0.18 * carbon_norm - 0.10 * econ_focus), 0.0, 1.0)
        summary = "EV charging prefers parking-window midday slots and scales back in carbon-intensive hours."
    else:
        raise ValueError(f"Unknown mock resource agent: {name}")
    return {"signal": series_payload(signal, digits=4), "summary": summary}


def _case1_mock_reconcile() -> dict[str, Any]:
    return {
        "ess_weight": 1.0,
        "hvac_weight": 1.0,
        "service_weight": 1.0,
        "ev_weight": 1.0,
        "summary": [
            "Weekly coordination keeps ESS as the main inter-day balancing device.",
            "HVAC and service demand should align with PV-rich workday noon periods.",
            "EV charging follows occupancy windows and avoids stacking into evening peaks.",
        ],
    }


class Case1MultiAgentOrchestrator:
    def __init__(self, model: str, use_mock_llm: bool, api_key: str | None = None):
        self.model = model
        self.use_mock_llm = use_mock_llm
        self.client = None if use_mock_llm else DeepSeekJSONClient(model=model, api_key=api_key or "")

    def generate_intent(self, scenario: dict[str, Any], run_index: int, trace_path: Path) -> dict[str, Any]:
        arrays = base_arrays(scenario)
        horizon = len(arrays["hours"])
        if self.use_mock_llm:
            operator = _case1_mock_operator(arrays)
            ess = _case1_mock_resource("ess_agent", arrays, operator)
            hvac = _case1_mock_resource("hvac_agent", arrays, operator)
            service = _case1_mock_resource("service_load_agent", arrays, operator)
            ev = _case1_mock_resource("ev_agent", arrays, operator)
            reconcile = _case1_mock_reconcile()
        else:
            operator = self._call_operator_agent(scenario, run_index)
            ess = self._call_resource_agent("ess_agent", scenario, run_index, operator)
            hvac = self._call_resource_agent("hvac_agent", scenario, run_index, operator)
            service = self._call_resource_agent("service_load_agent", scenario, run_index, operator)
            ev = self._call_resource_agent("ev_agent", scenario, run_index, operator)
            reconcile = self._call_reconcile_agent(run_index, operator, ess, hvac, service, ev)

        fallback_operator = _case1_mock_operator(arrays)
        fallback_ess = _case1_mock_resource("ess_agent", arrays, fallback_operator)
        fallback_hvac = _case1_mock_resource("hvac_agent", arrays, fallback_operator)
        fallback_service = _case1_mock_resource("service_load_agent", arrays, fallback_operator)
        fallback_ev = _case1_mock_resource("ev_agent", arrays, fallback_operator)

        operator_econ = _ensure_series(operator.get("econ_focus"), horizon, 0.0, 1.0, np.asarray(fallback_operator["econ_focus"], dtype=float))
        operator_carbon = _ensure_series(operator.get("carbon_focus"), horizon, 0.0, 1.0, np.asarray(fallback_operator["carbon_focus"], dtype=float))
        ess_signal = _ensure_series(ess.get("signal"), horizon, -1.0, 1.0, np.asarray(fallback_ess["signal"], dtype=float))
        hvac_signal = _ensure_series(hvac.get("signal"), horizon, -1.0, 1.0, np.asarray(fallback_hvac["signal"], dtype=float))
        service_signal = _ensure_series(service.get("signal"), horizon, -1.0, 1.0, np.asarray(fallback_service["signal"], dtype=float))
        ev_signal = _ensure_series(ev.get("signal"), horizon, 0.0, 1.0, np.asarray(fallback_ev["signal"], dtype=float))

        intent = {
            "operator_econ_focus": operator_econ,
            "operator_carbon_focus": operator_carbon,
            "ess_signal": np.clip(ess_signal * clamp(float(reconcile.get("ess_weight", 1.0)), 0.4, 1.6), -1.0, 1.0),
            "hvac_signal": np.clip(hvac_signal * clamp(float(reconcile.get("hvac_weight", 1.0)), 0.4, 1.6), -1.0, 1.0),
            "service_signal": np.clip(service_signal * clamp(float(reconcile.get("service_weight", 1.0)), 0.4, 1.6), -1.0, 1.0),
            "ev_signal": np.clip(ev_signal * clamp(float(reconcile.get("ev_weight", 1.0)), 0.4, 1.6), 0.0, 1.0),
            "summary": reconcile.get("summary", operator.get("summary", fallback_operator["summary"])),
            "agent_rationales": {
                "operator": operator.get("coordination_message", ""),
                "ess_agent": ess.get("summary", ""),
                "hvac_agent": hvac.get("summary", ""),
                "service_load_agent": service.get("summary", ""),
                "ev_agent": ev.get("summary", ""),
            },
            "model_name": self.model if not self.use_mock_llm else "mock_llm",
        }

        write_json(
            trace_path,
            {
                "run_index": run_index,
                "operator": operator,
                "ess_agent": ess,
                "hvac_agent": hvac,
                "service_load_agent": service,
                "ev_agent": ev,
                "reconcile_agent": reconcile,
                "combined_intent": {
                    key: series_payload(value, digits=4) if isinstance(value, np.ndarray) else value
                    for key, value in intent.items()
                },
            },
        )
        return intent

    def _call_operator_agent(self, scenario: dict[str, Any], run_index: int) -> dict[str, Any]:
        assert self.client is not None
        profiles = scenario["profiles"]
        config = scenario["config"]
        return self.client.generate_json(
            "You are the operator agent of a low-carbon commercial park. Return valid JSON only.",
            {
                "run_index": run_index,
                "horizon_hours": len(profiles["hours"]),
                "assets": {
                    "tie_line_kw": config["tie_line_limit_kw"],
                    "pv_kw": config["pv"]["rated_power_kw"],
                    "ess_power_kw": config["ess"]["rated_power_kw"],
                    "ess_energy_kwh": config["ess"]["energy_capacity_kwh"],
                },
                "signals": {
                    "buy_price": profiles["buy_price_rmb_per_kwh"],
                    "carbon_price": profiles["carbon_price_rmb_per_kg"],
                    "carbon_intensity": profiles["grid_carbon_intensity_kg_per_kwh"],
                    "pv_available": profiles["pv_available_kw"],
                },
                "output_schema": {
                    "econ_focus": "array in [0,1]",
                    "carbon_focus": "array in [0,1]",
                    "coordination_message": "one short sentence",
                    "summary": "3 short bullet-style strings",
                },
            },
        )

    def _call_resource_agent(self, agent_name: str, scenario: dict[str, Any], run_index: int, operator: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        profiles = scenario["profiles"]
        return self.client.generate_json(
            f"You are {agent_name} in a weekly low-carbon park dispatch problem. Return JSON only.",
            {
                "run_index": run_index,
                "agent": agent_name,
                "profiles": {
                    "hours": profiles["hours"],
                    "day_index": profiles["day_index"],
                    "hour_of_day": profiles["hour_of_day"],
                    "temperature": profiles["ambient_temperature_c"],
                    "buy_price": profiles["buy_price_rmb_per_kwh"],
                    "carbon_intensity": profiles["grid_carbon_intensity_kg_per_kwh"],
                    "pv_available": profiles["pv_available_kw"],
                    "hvac_baseline": profiles["flexible_loads_kw"]["hvac_load"],
                    "service_baseline": profiles["flexible_loads_kw"]["service_load"],
                    "ev_request": profiles["ev_energy_request_kwh"],
                },
                "operator_guidance": {
                    "econ_focus": operator.get("econ_focus"),
                    "carbon_focus": operator.get("carbon_focus"),
                    "coordination_message": operator.get("coordination_message", ""),
                },
                "output_schema": {"signal": "array", "summary": "one short sentence"},
            },
        )

    def _call_reconcile_agent(self, run_index: int, operator: dict[str, Any], ess: dict[str, Any], hvac: dict[str, Any], service: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        return self.client.generate_json(
            "You are the weekly coordination agent for a multi-agent park dispatch problem. Return JSON only.",
            {
                "run_index": run_index,
                "operator": operator,
                "ess_agent": ess,
                "hvac_agent": hvac,
                "service_load_agent": service,
                "ev_agent": ev,
                "output_schema": {
                    "ess_weight": "float in [0.4,1.6]",
                    "hvac_weight": "float in [0.4,1.6]",
                    "service_weight": "float in [0.4,1.6]",
                    "ev_weight": "float in [0.4,1.6]",
                    "summary": "3 short bullet-style strings",
                },
            },
        )


def _default_partner_priority(park_id: str) -> dict[str, float]:
    partner_priority = {other_id: 0.58 for other_id in CASE2_PARK_IDS}
    if park_id == "Park_A":
        partner_priority["Park_B"] = 0.86
        partner_priority["Park_C"] = 0.74
        partner_priority["Park_E"] = 0.78
    elif park_id == "Park_B":
        partner_priority["Park_A"] = 0.88
        partner_priority["Park_C"] = 0.72
        partner_priority["Park_D"] = 0.66
    elif park_id == "Park_C":
        partner_priority["Park_B"] = 0.82
        partner_priority["Park_A"] = 0.70
        partner_priority["Park_E"] = 0.80
    elif park_id == "Park_D":
        partner_priority["Park_A"] = 0.64
        partner_priority["Park_B"] = 0.64
        partner_priority["Park_E"] = 0.68
    elif park_id == "Park_E":
        partner_priority["Park_A"] = 0.82
        partner_priority["Park_C"] = 0.76
        partner_priority["Park_D"] = 0.70
    partner_priority.pop(park_id, None)
    return partner_priority


def _case2_mock_round_park_agent(
    park_id: str,
    park_type: str,
    state_arrays: dict[str, np.ndarray],
    carbon_sensitivity: float,
    hour_context: dict[str, Any],
    previous_round: dict[str, Any] | None,
    memory: dict[str, Any],
    round_index: int,
    max_rounds: int,
    subjective_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile = subjective_profile or {}
    theta_carbon = float(profile.get("carbon", 0.5))
    theta_risk = float(profile.get("risk", 0.5))
    theta_neg = float(profile.get("neg", 0.5))
    hour = int(hour_context["hour"])
    sell_cap = float(hour_context["base_sell_kw"])
    buy_cap = float(hour_context["base_buy_kw"])
    pv_export = float(hour_context["pv_export_kw"])
    buy_price = float(hour_context["buy_price_rmb_per_kwh"])
    sell_price = float(hour_context["sell_price_rmb_per_kwh"])
    carbon_position = float(hour_context.get("carbon_position_kg", 0.0))
    carbon_position_norm = float(hour_context.get("carbon_position_norm", 0.0))
    carbon_priority = clamp(
        0.28
        + 0.42 * float(hour_context["carbon_intensity_norm"])
        + 0.12 * carbon_sensitivity
        + 0.10 * max(-carbon_position_norm, 0.0)
        + 0.08 * float(memory.get("last_projection_gap_kw", 0.0) > 1e-6)
        + 0.18 * (theta_carbon - 0.5),
        0.0,
        1.0,
    )
    price_spread = max(buy_price - sell_price, 0.03)
    previous_feedback = previous_round.get("order_book_feedback", {}) if previous_round else {}
    gap_pressure = min(float(previous_feedback.get("bid_ask_gap_rmb_per_kwh", 0.0)) / price_spread, 1.0)
    round_pressure = round_index / max(max_rounds, 1)
    concession = clamp(
        0.16 + 0.34 * round_pressure + 0.10 * float(previous_round is not None) + 0.12 * gap_pressure
        + 0.20 * (theta_neg - 0.5)
        - 0.10 * (theta_risk - 0.5),
        0.0,
        1.0,
    )

    partner_priority = _default_partner_priority(park_id)
    if memory.get("last_counterparty") in partner_priority:
        partner_priority[str(memory["last_counterparty"])] = clamp(float(partner_priority[str(memory["last_counterparty"])]) + 0.08, 0.0, 1.0)

    if previous_round is not None:
        tentative_pairs = previous_round.get("tentative_pairs", [])
        for pair in tentative_pairs:
            seller = pair.get("seller")
            buyer = pair.get("buyer")
            if seller == park_id and buyer in partner_priority:
                partner_priority[str(buyer)] = clamp(float(partner_priority[str(buyer)]) + 0.10, 0.0, 1.0)
            if buyer == park_id and seller in partner_priority:
                partner_priority[str(seller)] = clamp(float(partner_priority[str(seller)]) + 0.10, 0.0, 1.0)

    export_willingness = clamp(
        0.16
        + 0.82 * min(sell_cap / max(float(hour_context["tie_line_limit_kw"]), 1.0), 1.0)
        + 0.10 * min(pv_export / max(float(hour_context["tie_line_limit_kw"]), 1.0), 1.0),
        0.0,
        1.0,
    )
    import_willingness = clamp(
        0.18
        + 0.78 * min(buy_cap / max(float(hour_context["tie_line_limit_kw"]), 1.0), 1.0)
        + 0.08 * float(hour_context["carbon_intensity_norm"]),
        0.0,
        1.0,
    )

    if park_type == "Renewable-Rich Park":
        export_willingness = clamp(export_willingness + 0.14, 0.0, 1.0)
    elif park_type == "Load-Intensive Park":
        import_willingness = clamp(import_willingness + 0.16, 0.0, 1.0)
    elif park_type == "Storage-Dominant Park":
        export_willingness = clamp(export_willingness + 0.06 * (1.0 - float(hour_context["daylight_flag"])), 0.0, 1.0)
        import_willingness = clamp(import_willingness + 0.06 * float(hour_context["carbon_intensity_norm"]), 0.0, 1.0)
    elif park_type == "Flexible-Demand Park":
        import_willingness = clamp(import_willingness + 0.10 * float(hour_context["carbon_intensity_norm"]), 0.0, 1.0)
        carbon_priority = clamp(carbon_priority + 0.08, 0.0, 1.0)

    carbon_market_posture = "balanced"
    if carbon_position > 250.0:
        carbon_market_posture = "credit_seller"
        export_willingness = clamp(export_willingness + 0.04, 0.0, 1.0)
    elif carbon_position < -250.0:
        carbon_market_posture = "credit_buyer"
        import_willingness = clamp(import_willingness + 0.04 * carbon_priority, 0.0, 1.0)

    if sell_cap >= buy_cap + 1e-6 and sell_cap > 1e-6:
        role = "seller"
    elif buy_cap > sell_cap + 1e-6 and buy_cap > 1e-6:
        role = "buyer"
    elif max(sell_cap, buy_cap) > 1e-6:
        role = "balanced"
    else:
        role = "idle"

    export_target = max(sell_cap * (0.80 + 0.30 * export_willingness), 0.0)
    import_target = max(buy_cap * (0.82 + 0.28 * import_willingness), 0.0)
    ask_price = round(float(sell_price + price_spread * (0.34 + 0.20 * (1.0 - concession))), 4)
    bid_price = round(float(buy_price - price_spread * (0.16 + 0.16 * concession)), 4)
    if role == "seller":
        import_target = 0.0
    elif role == "buyer":
        export_target = 0.0
    elif role == "idle":
        export_willingness = 0.0
        import_willingness = 0.0
        export_target = 0.0
        import_target = 0.0

    return {
        "role": role,
        "export_willingness": round(float(export_willingness), 4),
        "import_willingness": round(float(import_willingness), 4),
        "carbon_priority": round(float(carbon_priority), 4),
        "concession_factor": round(float(concession), 4),
        "export_target_kwh": round(float(export_target), 4),
        "import_target_kwh": round(float(import_target), 4),
        "ask_price_rmb_per_kwh": ask_price,
        "bid_price_rmb_per_kwh": bid_price,
        "carbon_market_posture": carbon_market_posture,
        "partner_priority": {other: round(float(value), 4) for other, value in partner_priority.items()},
        "continue_bidding": bool(round_index < max_rounds and role != "idle"),
        "message": f"{park_id} acts as {role} in round {round_index}, adjusting concession to {concession:.2f}.",
        "summary": f"{park_id} {role} stance: export {export_target:.1f} kWh, import {import_target:.1f} kWh.",
    }


def _build_order_book(
    park_outputs: dict[str, dict[str, Any]],
    round_index: int,
    max_rounds: int,
) -> dict[str, Any]:
    sell_orders: list[dict[str, Any]] = []
    buy_orders: list[dict[str, Any]] = []
    for park_id, payload in park_outputs.items():
        export_target = float(payload.get("export_target_kwh", 0.0))
        import_target = float(payload.get("import_target_kwh", 0.0))
        if export_target > 1e-6:
            sell_orders.append(
                {
                    "park_id": park_id,
                    "quantity_kwh": round(float(export_target), 4),
                    "ask_price_rmb_per_kwh": round(float(payload.get("ask_price_rmb_per_kwh", 0.0)), 4),
                    "carbon_market_posture": payload.get("carbon_market_posture", "balanced"),
                    "carbon_priority": round(float(payload.get("carbon_priority", 0.5)), 4),
                }
            )
        if import_target > 1e-6:
            buy_orders.append(
                {
                    "park_id": park_id,
                    "quantity_kwh": round(float(import_target), 4),
                    "bid_price_rmb_per_kwh": round(float(payload.get("bid_price_rmb_per_kwh", 0.0)), 4),
                    "carbon_market_posture": payload.get("carbon_market_posture", "balanced"),
                    "carbon_priority": round(float(payload.get("carbon_priority", 0.5)), 4),
                }
            )
    sell_orders.sort(key=lambda item: (float(item["ask_price_rmb_per_kwh"]), item["park_id"]))
    buy_orders.sort(key=lambda item: (-float(item["bid_price_rmb_per_kwh"]), item["park_id"]))
    return {
        "continue_bidding": bool(round_index < max_rounds),
        "sell_orders": sell_orders,
        "buy_orders": buy_orders,
        "summary": f"Order book receives {len(sell_orders)} sell orders and {len(buy_orders)} buy orders in round {round_index}.",
    }


def _double_auction_matches(order_book: dict[str, Any]) -> list[dict[str, Any]]:
    sellers = [dict(order) for order in order_book.get("sell_orders", [])]
    buyers = [dict(order) for order in order_book.get("buy_orders", [])]
    matches: list[dict[str, Any]] = []
    sell_idx = 0
    buy_idx = 0
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
        volume = min(float(seller_order["quantity_kwh"]), float(buyer_order["quantity_kwh"]))
        if volume <= 1e-6:
            if float(seller_order["quantity_kwh"]) <= 1e-6:
                sell_idx += 1
            if float(buyer_order["quantity_kwh"]) <= 1e-6:
                buy_idx += 1
            continue
        matches.append(
            {
                "seller": seller,
                "buyer": buyer,
                "volume_kwh": round(float(volume), 4),
                "price_rmb_per_kwh": round(float((ask + bid) / 2.0), 4),
                "ask_price_rmb_per_kwh": round(float(ask), 4),
                "bid_price_rmb_per_kwh": round(float(bid), 4),
            }
        )
        seller_order["quantity_kwh"] = float(seller_order["quantity_kwh"]) - volume
        buyer_order["quantity_kwh"] = float(buyer_order["quantity_kwh"]) - volume
        if float(seller_order["quantity_kwh"]) <= 1e-6:
            sell_idx += 1
        if float(buyer_order["quantity_kwh"]) <= 1e-6:
            buy_idx += 1
    return matches


def _order_book_feedback(
    order_book: dict[str, Any],
    tentative_pairs: list[dict[str, Any]],
    previous_round: dict[str, Any] | None,
) -> dict[str, Any]:
    sell_orders = order_book.get("sell_orders", [])
    buy_orders = order_book.get("buy_orders", [])
    matched_sellers = {str(pair["seller"]) for pair in tentative_pairs if float(pair.get("volume_kwh", 0.0)) > 1e-6}
    matched_buyers = {str(pair["buyer"]) for pair in tentative_pairs if float(pair.get("volume_kwh", 0.0)) > 1e-6}
    lowest_ask = min([float(order["ask_price_rmb_per_kwh"]) for order in sell_orders], default=0.0)
    highest_bid = max([float(order["bid_price_rmb_per_kwh"]) for order in buy_orders], default=0.0)
    bid_ask_gap = max(lowest_ask - highest_bid, 0.0) if sell_orders and buy_orders else 0.0
    previous_feedback = previous_round.get("order_book_feedback", {}) if previous_round else {}
    previous_gap = float(previous_feedback.get("bid_ask_gap_rmb_per_kwh", -1.0))
    matched_volume = sum(float(pair.get("volume_kwh", 0.0)) for pair in tentative_pairs)
    no_match = matched_volume <= 1e-6
    if not sell_orders or not buy_orders:
        failed_reason = "missing_order_side"
    elif bid_ask_gap > 1e-6:
        failed_reason = "bid_below_ask"
    elif no_match:
        failed_reason = "self_match_or_zero_quantity"
    else:
        failed_reason = ""
    return {
        "matched_volume_kwh": round(float(matched_volume), 4),
        "highest_bid_rmb_per_kwh": round(float(highest_bid), 4),
        "lowest_ask_rmb_per_kwh": round(float(lowest_ask), 4),
        "bid_ask_gap_rmb_per_kwh": round(float(bid_ask_gap), 4),
        "bid_ask_gap_stalled": bool(no_match and previous_gap >= 0.0 and abs(previous_gap - bid_ask_gap) <= 1e-4),
        "failed_match_reason": failed_reason,
        "unmatched_sell_orders": [order for order in sell_orders if str(order["park_id"]) not in matched_sellers],
        "unmatched_buy_orders": [order for order in buy_orders if str(order["park_id"]) not in matched_buyers],
    }


def _normalize_bidding_payload(
    park_id: str,
    raw: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    partner_priority = raw.get("partner_priority", fallback["partner_priority"])
    if not isinstance(partner_priority, dict):
        partner_priority = fallback["partner_priority"]
    role = str(raw.get("role", fallback["role"]))
    if role not in {"seller", "buyer", "balanced", "idle"}:
        role = fallback["role"]
    carbon_market_posture = str(raw.get("carbon_market_posture", fallback.get("carbon_market_posture", "balanced")))
    if carbon_market_posture not in {"credit_seller", "credit_buyer", "balanced"}:
        carbon_market_posture = fallback.get("carbon_market_posture", "balanced")
    return {
        "role": role,
        "export_willingness": round(float(clamp(float(raw.get("export_willingness", fallback["export_willingness"])), 0.0, 1.0)), 4),
        "import_willingness": round(float(clamp(float(raw.get("import_willingness", fallback["import_willingness"])), 0.0, 1.0)), 4),
        "carbon_priority": round(float(clamp(float(raw.get("carbon_priority", fallback["carbon_priority"])), 0.0, 1.0)), 4),
        "concession_factor": round(float(clamp(float(raw.get("concession_factor", fallback["concession_factor"])), 0.0, 1.0)), 4),
        "export_target_kwh": round(float(max(float(raw.get("export_target_kwh", fallback["export_target_kwh"])), 0.0)), 4),
        "import_target_kwh": round(float(max(float(raw.get("import_target_kwh", fallback["import_target_kwh"])), 0.0)), 4),
        "ask_price_rmb_per_kwh": round(float(max(float(raw.get("ask_price_rmb_per_kwh", fallback["ask_price_rmb_per_kwh"])), 0.0)), 4),
        "bid_price_rmb_per_kwh": round(float(max(float(raw.get("bid_price_rmb_per_kwh", fallback["bid_price_rmb_per_kwh"])), 0.0)), 4),
        "carbon_market_posture": carbon_market_posture,
        "partner_priority": {
            other_id: round(float(clamp(float(partner_priority.get(other_id, value)), 0.0, 1.0)), 4)
            for other_id, value in fallback["partner_priority"].items()
        },
        "continue_bidding": bool(raw.get("continue_bidding", raw.get("continue_negotiation", fallback["continue_bidding"]))),
        "message": str(raw.get("message", fallback["message"]))[:240],
        "summary": str(raw.get("summary", fallback["summary"]))[:240],
    }


class Case2MultiAgentOrchestrator:
    def __init__(self, model: str, use_mock_llm: bool, api_key: str | None = None):
        self.model = model
        self.use_mock_llm = use_mock_llm
        self.client = None if use_mock_llm else DeepSeekJSONClient(model=model, api_key=api_key or "")

    def generate_intent(self, states: dict[str, Any], run_index: int, trace_path: Path) -> dict[str, Any]:
        max_rounds = 5
        park_ids = list(states.keys())
        horizon = len(next(iter(states.values())).arrays["hours"])
        memory: dict[str, dict[str, Any]] = {
            park_id: {
                "last_role": "idle",
                "last_counterparty": "",
                "last_price": 0.0,
                "last_volume": 0.0,
                "last_projection_gap_kw": 0.0,
                "successful_hours": 0.0,
            }
            for park_id in park_ids
        }
        weekly_payload: dict[str, Any] = {
            "max_bidding_rounds": max_rounds,
            "information_boundary": (
                "In each round, park-market agents observe their private park states and heterogeneous profiles, "
                "public market signals, the public order book from the previous round, and feasibility feedback "
                "on their own submitted orders, but they do not access other parks' internal operating states "
                "or private preference parameters."
            ),
            "summary": [],
            "hours": {},
        }
        trace_payload: dict[str, Any] = {
            "run_index": run_index,
            "max_bidding_rounds": max_rounds,
            "information_boundary": weekly_payload["information_boundary"],
            "hours": {},
        }

        carbon_positions, max_abs_position = self._precompute_carbon_stats(states)
        for hour in range(horizon):
            hour_context = self._build_hour_context(states, hour, carbon_positions, max_abs_position)
            memory_before_hour = {park_id: dict(payload) for park_id, payload in memory.items()}
            if not hour_context["trade_potential"]:
                hour_summary = ["No bilateral trade potential in the feasible base dispatch for this hour."]
                weekly_payload["hours"][str(hour)] = {
                    "context": hour_context,
                    "rounds": [],
                    "hour_summary": hour_summary,
                }
                trace_payload["hours"][str(hour)] = {
                    "context": hour_context,
                    "memory_before_hour": memory_before_hour,
                    "rounds": [],
                    "hour_summary": hour_summary,
                    "memory_after_hour": memory,
                }
                continue
            previous_round: dict[str, Any] | None = None
            hour_rounds: list[dict[str, Any]] = []
            trace_rounds: list[dict[str, Any]] = []
            for round_index in range(1, max_rounds + 1):
                park_outputs: dict[str, dict[str, Any]] = {}
                park_raw: dict[str, Any] = {}
                fallback_outputs: dict[str, dict[str, Any]] = {}
                active_park_ids = set(hour_context["active_park_ids"])
                for park_id, state in states.items():
                    local_hour_context = {"hour": hour_context["hour"], **hour_context["parks"][park_id]}
                    fallback = _case2_mock_round_park_agent(
                        park_id=park_id,
                        park_type=state.spec.park_type,
                        state_arrays=state.arrays,
                        carbon_sensitivity=state.spec.carbon_sensitivity,
                        hour_context=local_hour_context,
                        previous_round=previous_round,
                        memory=memory[park_id],
                        round_index=round_index,
                        max_rounds=max_rounds,
                        subjective_profile=getattr(state, "subjective_profile", None),
                    )
                    fallback_outputs[park_id] = fallback

                if self.use_mock_llm:
                    for park_id, fallback in fallback_outputs.items():
                        raw_output = fallback
                        park_outputs[park_id] = _normalize_bidding_payload(park_id, raw_output, fallback)
                        park_raw[park_id] = raw_output
                else:
                    active_outputs = self._call_round_park_agents_parallel(
                        states=states,
                        run_index=run_index,
                        hour_context=hour_context,
                        previous_round=previous_round,
                        memory=memory,
                        round_index=round_index,
                        max_rounds=max_rounds,
                    )
                    for park_id, fallback in fallback_outputs.items():
                        raw_output = active_outputs.get(park_id, fallback)
                        park_outputs[park_id] = _normalize_bidding_payload(park_id, raw_output, fallback)
                        park_raw[park_id] = raw_output

                order_book = _build_order_book(park_outputs, round_index, max_rounds)
                tentative_pairs = _double_auction_matches(order_book)
                order_book_feedback = _order_book_feedback(order_book, tentative_pairs, previous_round)
                round_payload = {
                    "round_index": round_index,
                    "park_outputs": park_outputs,
                    "order_book": order_book,
                    "order_book_feedback": order_book_feedback,
                    "tentative_pairs": tentative_pairs,
                }
                hour_rounds.append(round_payload)
                trace_rounds.append(
                    {
                        "round_index": round_index,
                        "hour_context": hour_context,
                        "park_outputs": park_outputs,
                        "order_book": order_book,
                        "order_book_feedback": order_book_feedback,
                        "tentative_pairs": tentative_pairs,
                        "raw": {"parks": park_raw},
                    }
                )
                previous_round = {
                    "tentative_pairs": tentative_pairs,
                    "order_book_summary": order_book["summary"],
                    "order_book_feedback": order_book_feedback,
                }
                no_submitted_orders = not order_book["sell_orders"] or not order_book["buy_orders"]
                if (
                    not order_book["continue_bidding"]
                    or no_submitted_orders
                    or (round_index > 1 and order_book_feedback["bid_ask_gap_stalled"])
                ):
                    break

            memory = self._update_memory_from_hour(memory, hour_rounds)
            hour_summary = self._build_hour_summary(hour_rounds)
            weekly_payload["hours"][str(hour)] = {
                "context": hour_context,
                "rounds": hour_rounds,
                "hour_summary": hour_summary,
            }
            trace_payload["hours"][str(hour)] = {
                "context": hour_context,
                "memory_before_hour": memory_before_hour,
                "rounds": trace_rounds,
                "hour_summary": hour_summary,
                "memory_after_hour": {park_id: dict(payload) for park_id, payload in memory.items()},
            }

        weekly_payload["summary"] = self._build_week_summary(weekly_payload)
        trace_payload["summary"] = weekly_payload["summary"]
        write_json(trace_path, trace_payload)
        return weekly_payload

    @staticmethod
    def _precompute_carbon_stats(states: dict[str, Any]) -> tuple[dict[str, float], float]:
        weekly_loads = {
            park_id: float(np.sum(state.internal_run.details["hourly"]["total_load_kw"]))
            for park_id, state in states.items()
        }
        total_reference_emission = sum(float(state.internal_run.total_carbon_emission) for state in states.values())
        total_reference_load = sum(max(value, 0.0) for value in weekly_loads.values())
        benchmark_intensity = total_reference_emission / max(total_reference_load, 1e-6)
        carbon_positions = {
            park_id: weekly_loads[park_id] * benchmark_intensity * carbon_quota_factor(state.spec.park_type)
            - float(state.internal_run.total_carbon_emission)
            for park_id, state in states.items()
        }
        max_abs_position = max([abs(value) for value in carbon_positions.values()] + [1e-6])
        return carbon_positions, max_abs_position

    def _build_hour_context(
        self,
        states: dict[str, Any],
        hour: int,
        carbon_positions: dict[str, float],
        max_abs_position: float,
    ) -> dict[str, Any]:
        park_context = {}
        for park_id, state in states.items():
            hourly = state.internal_run.details["hourly"]
            park_context[park_id] = {
                "base_buy_kw": round(float(hourly["buy_kw"][hour]), 4),
                "base_sell_kw": round(float(hourly["sell_kw"][hour]), 4),
                "pv_export_kw": round(float(hourly["pv_export_kw"][hour]), 4),
                "load_kw": round(float(hourly["total_load_kw"][hour]), 4),
                "buy_price_rmb_per_kwh": round(float(state.arrays["buy_price"][hour]), 4),
                "sell_price_rmb_per_kwh": round(float(state.arrays["sell_price"][hour]), 4),
                "carbon_intensity": round(float(state.arrays["carbon_intensity"][hour]), 4),
                "carbon_price": round(float(state.arrays["carbon_price"][hour]), 4),
                "hour_of_day": int(state.arrays["hour_of_day"][hour]),
                "day_index": int(state.arrays["day_index"][hour]),
                "tie_line_limit_kw": round(float(state.scenario["config"]["tie_line_limit_kw"]), 4),
                "park_type": state.spec.park_type,
                "carbon_sensitivity": state.spec.carbon_sensitivity,
                "carbon_quota_kg": round(float(carbon_positions[park_id] + state.internal_run.total_carbon_emission), 4),
                "forecast_emission_kg": round(float(state.internal_run.total_carbon_emission), 4),
                "carbon_position_kg": round(float(carbon_positions[park_id]), 4),
                "carbon_position_norm": round(float(carbon_positions[park_id] / max_abs_position), 4),
            }
        carbon_values = [payload["carbon_intensity"] for payload in park_context.values()]
        max_carbon = max(max(carbon_values), 1e-6)
        daylight_flag = 1.0 if any(9 <= payload["hour_of_day"] <= 17 for payload in park_context.values()) else 0.0
        for payload in park_context.values():
            payload["carbon_intensity_norm"] = round(float(payload["carbon_intensity"] / max_carbon), 4)
            payload["daylight_flag"] = daylight_flag
        seller_ids = [park_id for park_id, payload in park_context.items() if payload["base_sell_kw"] > 1e-6]
        buyer_ids = [park_id for park_id, payload in park_context.items() if payload["base_buy_kw"] > 1e-6]
        active_park_ids = sorted(set(seller_ids) | set(buyer_ids))
        return {
            "hour": hour,
            "parks": park_context,
            "seller_ids": seller_ids,
            "buyer_ids": buyer_ids,
            "active_park_ids": active_park_ids,
            "trade_potential": bool(seller_ids and buyer_ids),
        }

    def _call_round_park_agents_parallel(
        self,
        states: dict[str, Any],
        run_index: int,
        hour_context: dict[str, Any],
        previous_round: dict[str, Any] | None,
        memory: dict[str, dict[str, Any]],
        round_index: int,
        max_rounds: int,
    ) -> dict[str, dict[str, Any]]:
        assert self.client is not None
        active_park_ids = list(hour_context["active_park_ids"])
        if not active_park_ids:
            return {}
        outputs: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(3, len(active_park_ids))) as executor:
            future_map = {
                executor.submit(
                    self._call_round_park_agent,
                    park_id=park_id,
                    state=states[park_id],
                    states=states,
                    run_index=run_index,
                    hour_context=hour_context,
                    previous_round=previous_round,
                    memory=memory[park_id],
                    round_index=round_index,
                    max_rounds=max_rounds,
                ): park_id
                for park_id in active_park_ids
            }
            for future in as_completed(future_map):
                park_id = future_map[future]
                outputs[park_id] = future.result()
        return outputs

    def _call_round_park_agent(
        self,
        park_id: str,
        state: Any,
        states: dict[str, Any],
        run_index: int,
        hour_context: dict[str, Any],
        previous_round: dict[str, Any] | None,
        memory: dict[str, Any],
        round_index: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        assert self.client is not None
        # Keep the bidding setting decentralized: the agent sees its private state
        # and public order-book feedback, not other parks' internal operating states.
        return self.client.generate_json(
            f"You are the bidding agent for {park_id} in a multi-round inter-park electricity trading problem. Return JSON only.",
            {
                "run_index": run_index,
                "park_id": park_id,
                "park_type": state.spec.park_type,
                "carbon_sensitivity": state.spec.carbon_sensitivity,
                "subjective_profile": getattr(state, "subjective_profile", None) or {},
                "profile_rationales": getattr(state, "profile_rationales", None) or {},
                "round_context": {
                    "hour": hour_context["hour"],
                    "round_index": round_index,
                    "max_rounds": max_rounds,
                    "local_state": hour_context["parks"][park_id],
                    "public_market": {
                        "active_park_ids": hour_context["active_park_ids"],
                        "seller_ids": hour_context["seller_ids"],
                        "buyer_ids": hour_context["buyer_ids"],
                    },
                    "public_order_book_feedback": (previous_round or {}).get("order_book_feedback", {}),
                    "memory": memory,
                    "previous_round": previous_round or {},
                },
                "output_schema": {
                    "role": "seller|buyer|balanced|idle",
                    "export_willingness": "float in [0,1]",
                    "import_willingness": "float in [0,1]",
                    "carbon_priority": "float in [0,1]",
                    "concession_factor": "float in [0,1]",
                    "export_target_kwh": "non-negative float",
                    "import_target_kwh": "non-negative float",
                    "ask_price_rmb_per_kwh": "non-negative float",
                    "bid_price_rmb_per_kwh": "non-negative float",
                    "carbon_market_posture": "credit_seller|credit_buyer|balanced",
                    "partner_priority": "object keyed by other parks with floats in [0,1]",
                    "continue_bidding": "boolean",
                    "message": "one short sentence",
                    "summary": "one short sentence",
                },
            },
        )

    def _update_memory_from_hour(self, memory: dict[str, dict[str, Any]], hour_rounds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        updated = {park_id: dict(payload) for park_id, payload in memory.items()}
        if not hour_rounds:
            return updated
        final_round = hour_rounds[-1]
        matched_counterparty: dict[str, str] = {}
        matched_price: dict[str, float] = {}
        matched_volume: dict[str, float] = {}
        for pair in final_round.get("tentative_pairs", []):
            seller = str(pair["seller"])
            buyer = str(pair["buyer"])
            matched_counterparty[seller] = buyer
            matched_counterparty[buyer] = seller
            matched_price[seller] = float(pair["price_rmb_per_kwh"])
            matched_price[buyer] = float(pair["price_rmb_per_kwh"])
            matched_volume[seller] = float(pair["volume_kwh"])
            matched_volume[buyer] = float(pair["volume_kwh"])
        for park_id, park_output in final_round.get("park_outputs", {}).items():
            updated[park_id]["last_role"] = park_output["role"]
            updated[park_id]["last_counterparty"] = matched_counterparty.get(park_id, "")
            updated[park_id]["last_price"] = round(float(matched_price.get(park_id, park_output["ask_price_rmb_per_kwh"])), 4)
            updated[park_id]["last_volume"] = round(float(matched_volume.get(park_id, 0.0)), 4)
            updated[park_id]["last_projection_gap_kw"] = 0.0
            if matched_volume.get(park_id, 0.0) > 1e-6:
                updated[park_id]["successful_hours"] = float(updated[park_id].get("successful_hours", 0.0)) + 1.0
        return updated

    def _build_hour_summary(self, hour_rounds: list[dict[str, Any]]) -> list[str]:
        if not hour_rounds:
            return ["No bidding rounds were executed."]
        final_round = hour_rounds[-1]
        pair_text = ", ".join(
            f"{item['seller']}->{item['buyer']} ({item['volume_kwh']:.1f} kWh)"
            for item in final_round.get("tentative_pairs", [])
        )
        if not pair_text:
            pair_text = "no tentative agreement"
        return [
            f"Rounds executed: {len(hour_rounds)}.",
            f"Final tentative pairs: {pair_text}.",
            f"Order-book summary: {final_round['order_book']['summary']}",
        ]

    def _build_week_summary(self, weekly_payload: dict[str, Any]) -> list[str]:
        active_hours = 0
        multi_round_hours = 0
        matched_hours = 0
        for hour_payload in weekly_payload["hours"].values():
            rounds = hour_payload.get("rounds", [])
            if rounds:
                active_hours += 1
            if len(rounds) > 1:
                multi_round_hours += 1
            if rounds and rounds[-1].get("tentative_pairs"):
                matched_hours += 1
        return [
            f"Active bidding hours: {active_hours}.",
            f"Multi-round hours: {multi_round_hours}.",
            f"Hours with tentative agreements: {matched_hours}.",
        ]
