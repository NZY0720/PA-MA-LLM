"""Independent PPO baseline for Case II inter-park bidding.

Implementation notes
--------------------
Because the bundled Python distribution has only numpy available, the actor
and critic networks are kept linear (state -> action mean / value) with a
learned per-action log-standard-deviation vector. Each park has its own
independent policy and critic; there is no centralised critic and no
communication beyond what the environment makes public (same observation
boundary as the proposed framework). PPO is implemented with GAE(lambda),
clipped surrogate loss, an entropy bonus, and a value-function loss.

The trained policy is rolled out on the same 168-hour scenario used by the
LLM baselines. Each park submits orders (export_target_kwh,
import_target_kwh, ask_price, bid_price) at every hour; orders are matched
by the same double-auction rule used by the LLM baselines and then passed
through the physics-aware clearing layer, so the action consequence
function is identical across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from methods.common import CASE2_PARK_IDS


OBS_FIELDS = (
    "load_norm",
    "pv_export_norm",
    "base_buy_norm",
    "base_sell_norm",
    "buy_price_norm",
    "sell_price_norm",
    "carbon_intensity_norm",
    "carbon_price_norm",
    "carbon_position_norm",
    "tie_line_norm",
    "soc_norm",
    "hour_phase_sin",
    "hour_phase_cos",
)
N_OBS = len(OBS_FIELDS)
N_ACT = 4  # export_kwh, import_kwh, ask, bid


@dataclass
class PolicyParams:
    W: np.ndarray   # (N_OBS, N_ACT)
    b: np.ndarray   # (N_ACT,)
    log_std: np.ndarray  # (N_ACT,)


@dataclass
class ValueParams:
    w: np.ndarray   # (N_OBS,)
    b: float


@dataclass
class MARLConfig:
    episodes: int = 400
    learning_rate: float = 3e-3
    value_lr: float = 1e-2
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    gamma: float = 0.95
    lam: float = 0.9
    update_epochs: int = 4
    seed: int = 20260512


def _init_policy(rng: np.random.Generator) -> PolicyParams:
    W = rng.normal(0.0, 0.10, size=(N_OBS, N_ACT))
    b = np.zeros(N_ACT)
    log_std = np.full(N_ACT, -0.7)
    return PolicyParams(W=W, b=b, log_std=log_std)


def _init_value(rng: np.random.Generator) -> ValueParams:
    return ValueParams(w=rng.normal(0.0, 0.05, size=N_OBS), b=0.0)


def _obs(state: Any, hour: int) -> np.ndarray:
    arrays = state.arrays
    spec = state.spec
    horizon = len(arrays["hours"])
    tie = float(spec.tie_line_scale) * 1.0
    load = float(arrays["inflexible"][hour] + arrays["hvac_baseline"][hour] + arrays["service_baseline"][hour] + arrays["ev_request"][hour]) / 600.0
    pv_export = float(arrays["pv"][hour]) / 600.0
    base_buy = float(state.internal_run.details["hourly"]["buy_kw"][hour]) / 600.0
    base_sell = float(state.internal_run.details["hourly"]["sell_kw"][hour]) / 600.0
    buy_price = float(arrays["buy_price"][hour]) / 1.2
    sell_price = float(arrays["sell_price"][hour]) / 1.2
    carbon_intensity = float(arrays["carbon_intensity"][hour]) / 0.8
    carbon_price = float(arrays["carbon_price"][hour]) / 0.3
    soc = float(state.internal_run.details["hourly"]["soc_kwh"][hour]) / max(float(state.scenario["config"]["ess"]["energy_capacity_kwh"]), 1.0)
    phase = 2.0 * np.pi * hour / max(horizon, 1)
    obs = np.array(
        [
            load,
            pv_export,
            base_buy,
            base_sell,
            buy_price,
            sell_price,
            carbon_intensity,
            carbon_price,
            0.0,  # carbon_position_norm filled in below if available
            tie,
            soc,
            float(np.sin(phase)),
            float(np.cos(phase)),
        ],
        dtype=float,
    )
    return obs


def _act_logp(policy: PolicyParams, obs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    mean = obs @ policy.W + policy.b
    std = np.exp(policy.log_std)
    raw = rng.normal(mean, std)
    diff = (raw - mean) / std
    logp = float(-0.5 * (diff * diff).sum() - policy.log_std.sum() - 0.5 * N_ACT * np.log(2.0 * np.pi))
    return raw, mean, logp


def _logp_under(policy: PolicyParams, obs_batch: np.ndarray, action_batch: np.ndarray) -> np.ndarray:
    mean = obs_batch @ policy.W + policy.b
    std = np.exp(policy.log_std)
    diff = (action_batch - mean) / std
    return -0.5 * (diff * diff).sum(axis=1) - policy.log_std.sum() - 0.5 * N_ACT * np.log(2.0 * np.pi)


def _entropy(policy: PolicyParams) -> float:
    return float(0.5 * N_ACT * (np.log(2.0 * np.pi) + 1.0) + policy.log_std.sum())


def _squash_action(raw: np.ndarray, base_buy: float, base_sell: float, buy_price: float, sell_price: float) -> dict[str, float]:
    """Map raw real-valued action to a feasible bidding order via tanh squashing.

    Quantity ranges are scaled relative to a fixed reference (180 kWh) so
    that the policy can propose non-trivial trades even when the
    physics-aware baseline reports zero net export/import for the hour;
    the production auction subsequently clips back to the realised
    flexibility headroom.
    """
    sq = np.tanh(raw)
    export_target = max(0.0, 90.0 * (sq[0] + 1.0))   # ~[0, 180]
    import_target = max(0.0, 90.0 * (sq[1] + 1.0))   # ~[0, 180]
    spread = max(buy_price - sell_price, 0.05)
    ask = sell_price + 0.5 * spread * (sq[2] + 1.0)  # ~[sell_price, buy_price]
    bid = sell_price + 0.5 * spread * (sq[3] + 1.0)
    return {
        "export_target_kwh": round(float(export_target), 4),
        "import_target_kwh": round(float(import_target), 4),
        "ask_price_rmb_per_kwh": round(float(ask), 4),
        "bid_price_rmb_per_kwh": round(float(bid), 4),
    }


def _rollout_one_episode(
    policies: dict[str, PolicyParams],
    values: dict[str, ValueParams],
    states: dict[str, Any],
    rng: np.random.Generator,
    *,
    horizon: int,
    deterministic: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Sample one episode of inter-park bidding under independent policies.

    Reward design (per park, per hour):
        r = (avoided grid carbon cost) + (positive trade margin)
          - (over-bid penalty if the order is infeasible vs base_sell/base_buy)
    """
    rollouts: dict[str, list[dict[str, Any]]] = {p: [] for p in states}
    park_ids = list(states.keys())
    rewards: dict[str, list[float]] = {p: [] for p in park_ids}

    for hour in range(horizon):
        obs_by_park = {p: _obs(states[p], hour) for p in park_ids}
        action_by_park: dict[str, dict[str, float]] = {}
        raw_by_park: dict[str, np.ndarray] = {}
        logp_by_park: dict[str, float] = {}
        mean_by_park: dict[str, np.ndarray] = {}
        for p in park_ids:
            arrays = states[p].arrays
            details = states[p].internal_run.details["hourly"]
            base_buy = float(details["buy_kw"][hour])
            base_sell = float(details["sell_kw"][hour])
            buy_price = float(arrays["buy_price"][hour])
            sell_price = float(arrays["sell_price"][hour])
            if deterministic:
                raw = obs_by_park[p] @ policies[p].W + policies[p].b
                logp = 0.0
                mean = raw
            else:
                raw, mean, logp = _act_logp(policies[p], obs_by_park[p], rng)
            raw_by_park[p] = raw
            logp_by_park[p] = logp
            mean_by_park[p] = mean
            action_by_park[p] = _squash_action(raw, base_buy, base_sell, buy_price, sell_price)

        sell_orders = []
        buy_orders = []
        for p, action in action_by_park.items():
            if action["export_target_kwh"] > 1e-3:
                sell_orders.append({"park_id": p, "ask": action["ask_price_rmb_per_kwh"], "qty": action["export_target_kwh"]})
            if action["import_target_kwh"] > 1e-3:
                buy_orders.append({"park_id": p, "bid": action["bid_price_rmb_per_kwh"], "qty": action["import_target_kwh"]})
        sell_orders.sort(key=lambda o: o["ask"])
        buy_orders.sort(key=lambda o: -o["bid"])

        accepted_volume = {p: 0.0 for p in park_ids}
        accepted_revenue = {p: 0.0 for p in park_ids}
        accepted_cost = {p: 0.0 for p in park_ids}
        i = j = 0
        while i < len(sell_orders) and j < len(buy_orders):
            s = sell_orders[i]
            b = buy_orders[j]
            if s["park_id"] == b["park_id"]:
                j += 1
                continue
            if b["bid"] + 1e-9 < s["ask"]:
                break
            vol = min(s["qty"], b["qty"])
            price = 0.5 * (s["ask"] + b["bid"])
            accepted_volume[s["park_id"]] += vol
            accepted_volume[b["park_id"]] += vol
            accepted_revenue[s["park_id"]] += vol * price
            accepted_cost[b["park_id"]] += vol * price
            s["qty"] -= vol
            b["qty"] -= vol
            if s["qty"] <= 1e-6:
                i += 1
            if b["qty"] <= 1e-6:
                j += 1

        for p in park_ids:
            arrays = states[p].arrays
            details = states[p].internal_run.details["hourly"]
            base_buy = float(details["buy_kw"][hour])
            base_sell = float(details["sell_kw"][hour])
            buy_price = float(arrays["buy_price"][hour])
            sell_price = float(arrays["sell_price"][hour])
            grid_intensity = float(arrays["carbon_intensity"][hour])
            grid_carbon_price = float(arrays["carbon_price"][hour])
            margin = (accepted_revenue[p] - sell_price * accepted_volume[p]) if accepted_revenue[p] > 0 else 0.0
            margin += (buy_price * accepted_volume[p] - accepted_cost[p]) if accepted_cost[p] > 0 else 0.0
            avoided_carbon = grid_carbon_price * grid_intensity * max(accepted_cost[p] / max(buy_price, 1e-6), 0.0)
            over_export = max(action_by_park[p]["export_target_kwh"] - base_sell, 0.0)
            over_import = max(action_by_park[p]["import_target_kwh"] - base_buy, 0.0)
            penalty = 0.40 * over_export + 0.40 * over_import
            reward = 0.01 * (margin + avoided_carbon) - penalty
            rewards[p].append(reward)
            rollouts[p].append(
                {
                    "obs": obs_by_park[p],
                    "raw_action": raw_by_park[p],
                    "logp": logp_by_park[p],
                    "reward": reward,
                    "mean": mean_by_park[p],
                    "accepted_volume": accepted_volume[p],
                    "action": action_by_park[p],
                }
            )

    rewards_arr = {p: np.array(rewards[p], dtype=float) for p in park_ids}
    return rollouts, rewards_arr


def _compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    policy: PolicyParams,
    value: ValueParams,
    obs: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    config: MARLConfig,
) -> None:
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-6
    adv_norm = (advantages - adv_mean) / adv_std
    for _ in range(config.update_epochs):
        # Policy update via PPO-clip
        mean = obs @ policy.W + policy.b
        std = np.exp(policy.log_std)
        diff = (actions - mean) / std
        logp = -0.5 * (diff * diff).sum(axis=1) - policy.log_std.sum() - 0.5 * N_ACT * np.log(2.0 * np.pi)
        ratio = np.exp(logp - old_logp)
        clipped = np.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
        # Use the negative of the (unclipped) surrogate to align gradient direction
        surrogate_term = np.minimum(ratio * adv_norm, clipped * adv_norm)
        # Gradient of logp wrt mean = diff / std (per-action component)
        # We approximate gradient ascent by summing weighted gradients
        weight = np.where(
            ratio * adv_norm <= clipped * adv_norm,
            ratio,
            np.where(adv_norm >= 0.0, clipped, ratio),
        )
        weight = weight * adv_norm  # scalar per sample
        grad_mean = (diff / std) * weight[:, None]
        grad_W = obs.T @ grad_mean
        grad_b = grad_mean.sum(axis=0)
        # Entropy gradient (encourages exploration) - log_std component
        grad_log_std = (diff * diff - 1.0).sum(axis=0) * (weight.mean()) + config.entropy_coef
        policy.W += config.learning_rate * grad_W / max(len(obs), 1)
        policy.b += config.learning_rate * grad_b / max(len(obs), 1)
        policy.log_std += 0.5 * config.learning_rate * grad_log_std / max(len(obs), 1)
        policy.log_std = np.clip(policy.log_std, -2.5, 0.5)

        # Value update via simple MSE gradient step
        v_pred = obs @ value.w + value.b
        v_err = returns - v_pred
        value.w += config.value_lr * (obs.T @ v_err) / max(len(obs), 1)
        value.b += float(config.value_lr * v_err.mean())


def train_independent_ppo(
    states: dict[str, Any],
    config: MARLConfig | None = None,
) -> dict[str, PolicyParams]:
    cfg = config or MARLConfig()
    rng = np.random.default_rng(cfg.seed)
    park_ids = [p for p in CASE2_PARK_IDS if p in states]
    horizon = len(next(iter(states.values())).arrays["hours"])
    policies = {p: _init_policy(rng) for p in park_ids}
    values = {p: _init_value(rng) for p in park_ids}
    for episode in range(cfg.episodes):
        rollouts, rewards = _rollout_one_episode(policies, values, states, rng, horizon=horizon)
        for p in park_ids:
            obs_arr = np.stack([step["obs"] for step in rollouts[p]])
            act_arr = np.stack([step["raw_action"] for step in rollouts[p]])
            old_logp = np.array([step["logp"] for step in rollouts[p]])
            v_pred = obs_arr @ values[p].w + values[p].b
            advantages, returns = _compute_gae(rewards[p], v_pred, cfg.gamma, cfg.lam)
            _ppo_update(policies[p], values[p], obs_arr, act_arr, old_logp, advantages, returns, cfg)
    return policies


def rollout_to_intent(
    policies: dict[str, PolicyParams],
    states: dict[str, Any],
    *,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Run a deterministic rollout under trained policies and emit a
    case2-compatible intent payload."""
    park_ids = list(states.keys())
    horizon = horizon or len(next(iter(states.values())).arrays["hours"])
    rng = np.random.default_rng(0)
    rollouts, _ = _rollout_one_episode(policies, {p: _init_value(rng) for p in park_ids}, states, rng, horizon=horizon, deterministic=True)

    weekly_hours: dict[str, Any] = {}
    for hour in range(horizon):
        park_outputs: dict[str, dict[str, Any]] = {}
        for p in park_ids:
            step = rollouts[p][hour]
            action = step["action"]
            role = "balanced"
            if action["export_target_kwh"] > action["import_target_kwh"] + 1e-3:
                role = "seller"
            elif action["import_target_kwh"] > action["export_target_kwh"] + 1e-3:
                role = "buyer"
            elif action["export_target_kwh"] < 1e-3 and action["import_target_kwh"] < 1e-3:
                role = "idle"
            park_outputs[p] = {
                "role": role,
                "export_target_kwh": action["export_target_kwh"],
                "import_target_kwh": action["import_target_kwh"],
                "ask_price_rmb_per_kwh": action["ask_price_rmb_per_kwh"],
                "bid_price_rmb_per_kwh": action["bid_price_rmb_per_kwh"],
                "carbon_priority": 0.5,
                "concession_factor": 0.5,
                "carbon_market_posture": "balanced",
                "partner_priority": {q: 0.5 for q in park_ids if q != p},
                "continue_bidding": False,
                "message": "marl_policy",
                "summary": "marl_policy",
                "export_willingness": 1.0 if action["export_target_kwh"] > 1e-3 else 0.0,
                "import_willingness": 1.0 if action["import_target_kwh"] > 1e-3 else 0.0,
            }
        # Do NOT pre-build the order_book here — the production
        # _run_llm_trading rebuilds it from park_outputs if "order_book"
        # is absent, which is what we want.
        weekly_hours[str(hour)] = {
            "context": {"hour": hour, "active_park_ids": park_ids, "seller_ids": [], "buyer_ids": []},
            "rounds": [
                {
                    "round_index": 1,
                    "park_outputs": park_outputs,
                    "order_book_feedback": {},
                    "tentative_pairs": [],
                }
            ],
            "hour_summary": ["marl_policy"],
        }
    return {
        "max_bidding_rounds": 1,
        "information_boundary": "Independent PPO policies trained on decentralised observations.",
        "summary": ["independent_ppo"],
        "hours": weekly_hours,
        "model_name": "independent_ppo",
    }
