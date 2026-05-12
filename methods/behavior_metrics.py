"""Behavior-fidelity metrics for Case II.

Four metrics are computed for each Case II method (B2/B3/B4'/B5/MARL/...)
from its weekly bidding intent and the park profiles:

  RPA  Role-Profile Alignment
       Fraction of business-hour intervals (with at least one active role)
       in which the park's dominant role matches the role expected from its
       park_type.

  BHI  Bid Heterogeneity Index
       Mean pairwise Euclidean distance between per-park behavior signature
       vectors. Higher means parks behave more distinctly.

  PBDC Profile-Behavior Distance Correlation
       Pearson correlation between pairwise distances of profile vectors
       and pairwise distances of behavior signature vectors. Positive
       values indicate that parks with similar profiles also behave
       similarly.

  IDA  Identifiability Accuracy
       Leave-one-out 1-NN top-1 accuracy on per-hour action vectors with
       cosine similarity. Random baseline = 1/n_parks (= 0.20 for five
       parks). A value substantially above 0.20 indicates that each park
       has a distinguishable behavioral fingerprint.

Inputs are taken from the orchestrator-produced intent payload and the
ParkState objects, so these metrics can be computed for any baseline that
produces a compatible intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EXPECTED_ROLE = {
    "Renewable-Rich Park": "seller",
    "Load-Intensive Park": "buyer",
    "Storage-Dominant Park": "balanced",
    "Standard Park": "balanced",
    "Flexible-Demand Park": "buyer",
}

_BEHAVIOR_FIELDS = (
    "role",
    "export_target_kwh",
    "import_target_kwh",
    "ask_price_rmb_per_kwh",
    "bid_price_rmb_per_kwh",
    "carbon_priority",
    "concession_factor",
)


@dataclass
class BehaviorFidelity:
    rpa: float
    bhi: float
    pbdc: float
    ida: float

    def as_dict(self) -> dict[str, float]:
        return {
            "role_profile_alignment": round(self.rpa, 4),
            "bid_heterogeneity_index": round(self.bhi, 4),
            "profile_behavior_distance_correlation": round(self.pbdc, 4),
            "identifiability_accuracy": round(self.ida, 4),
        }


def _final_round_outputs(intent: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return {hour: park_outputs} from the final attempted round of each hour."""
    hours = intent.get("hours", {})
    out: dict[int, dict[str, Any]] = {}
    for hour_str, hour_payload in hours.items():
        rounds = hour_payload.get("rounds", [])
        if not rounds:
            continue
        out[int(hour_str)] = rounds[-1].get("park_outputs", {})
    return out


def _role_to_code(role: str) -> int:
    return {"seller": 0, "buyer": 1, "balanced": 2, "idle": 3}.get(str(role), 3)


def _park_hourly_vectors(
    park_ids: list[str],
    final_outputs: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Return per-park (n_hours, n_features) action arrays.

    Only hours where the park is not idle are included; if a park has no
    active hour the feature matrix is shape (0, F).
    """
    out: dict[str, list[list[float]]] = {park_id: [] for park_id in park_ids}
    for _, park_outputs in sorted(final_outputs.items()):
        for park_id in park_ids:
            payload = park_outputs.get(park_id)
            if not payload:
                continue
            role = str(payload.get("role", "idle"))
            if role == "idle":
                continue
            out[park_id].append(
                [
                    1.0 if role == "seller" else 0.0,
                    1.0 if role == "buyer" else 0.0,
                    1.0 if role == "balanced" else 0.0,
                    float(payload.get("export_target_kwh", 0.0)),
                    float(payload.get("import_target_kwh", 0.0)),
                    float(payload.get("ask_price_rmb_per_kwh", 0.0)),
                    float(payload.get("bid_price_rmb_per_kwh", 0.0)),
                    float(payload.get("carbon_priority", 0.0)),
                    float(payload.get("concession_factor", 0.0)),
                ]
            )
    return {park_id: np.array(rows, dtype=float) if rows else np.zeros((0, 9)) for park_id, rows in out.items()}


def _park_signature(action_matrix: np.ndarray) -> np.ndarray:
    if action_matrix.shape[0] == 0:
        return np.zeros(action_matrix.shape[1])
    return action_matrix.mean(axis=0)


def _park_profile_vector(state: Any) -> np.ndarray:
    """Concatenate hard spec scales + soft theta values into one profile vector."""
    spec = state.spec
    hard = np.array(
        [
            float(spec.pv_scale),
            float(spec.ess_power_scale),
            float(spec.ess_energy_scale),
            float(spec.inflexible_scale),
            float(spec.hvac_scale),
            float(spec.service_scale),
            float(spec.ev_scale),
            float(spec.tie_line_scale),
            float(spec.carbon_sensitivity),
        ],
        dtype=float,
    )
    theta = getattr(state, "subjective_profile", None) or {}
    soft = np.array(
        [
            float(theta.get("risk", 0.5)),
            float(theta.get("carbon", 0.5)),
            float(theta.get("service", 0.5)),
            float(theta.get("autonomy", 0.5)),
            float(theta.get("fair", 0.5)),
            float(theta.get("neg", 0.5)),
        ],
        dtype=float,
    )
    return np.concatenate([hard, soft])


def _pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    n = vectors.shape[0]
    if n < 2:
        return np.zeros(0)
    diffs = vectors[:, None, :] - vectors[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    return dists[iu]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def _role_profile_alignment(park_ids: list[str], states: dict[str, Any], final_outputs: dict[int, dict[str, Any]]) -> float:
    correct = 0
    total = 0
    for park_id in park_ids:
        expected = EXPECTED_ROLE.get(states[park_id].spec.park_type, "balanced")
        role_counts: dict[str, int] = {"seller": 0, "buyer": 0, "balanced": 0}
        for _, park_outputs in final_outputs.items():
            payload = park_outputs.get(park_id)
            if not payload:
                continue
            role = str(payload.get("role", "idle"))
            if role in role_counts:
                role_counts[role] += 1
        dominant = max(role_counts, key=lambda r: role_counts[r]) if any(role_counts.values()) else "idle"
        total += 1
        if dominant == expected:
            correct += 1
    return correct / max(total, 1)


def _bid_heterogeneity_index(signatures: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between per-park signatures.

    Signatures are standardised per feature before distance computation so
    that no field dominates the metric purely because of unit scale.
    """
    if signatures.shape[0] < 2:
        return 0.0
    std = signatures.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    normalised = (signatures - signatures.mean(axis=0)) / std
    dists = _pairwise_distances(normalised)
    return float(dists.mean()) if dists.size else 0.0


def _profile_behavior_distance_correlation(profile_vecs: np.ndarray, signatures: np.ndarray) -> float:
    if profile_vecs.shape[0] < 3 or signatures.shape[0] < 3:
        return 0.0
    sig_std = signatures.std(axis=0)
    sig_std = np.where(sig_std < 1e-9, 1.0, sig_std)
    sig_norm = (signatures - signatures.mean(axis=0)) / sig_std
    profile_dists = _pairwise_distances(profile_vecs)
    behavior_dists = _pairwise_distances(sig_norm)
    return _pearson(profile_dists, behavior_dists)


def _identifiability_accuracy(action_by_park: dict[str, np.ndarray]) -> float:
    parks = list(action_by_park.keys())
    park_to_idx = {p: i for i, p in enumerate(parks)}
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for park_id, matrix in action_by_park.items():
        if matrix.shape[0] == 0:
            continue
        for row in matrix:
            vectors.append(row)
            labels.append(park_to_idx[park_id])
    if len(vectors) < 2 or len(set(labels)) < 2:
        return 1.0 / max(len(parks), 1)
    X = np.array(vectors)
    y = np.array(labels)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    correct = int((y[nn] == y).sum())
    return correct / len(vectors)


def compute_behavior_fidelity(
    states: dict[str, Any],
    intent: dict[str, Any],
) -> BehaviorFidelity:
    park_ids = list(states.keys())
    final_outputs = _final_round_outputs(intent)
    action_by_park = _park_hourly_vectors(park_ids, final_outputs)
    signatures = np.stack([_park_signature(action_by_park[p]) for p in park_ids])
    profiles = np.stack([_park_profile_vector(states[p]) for p in park_ids])

    rpa = _role_profile_alignment(park_ids, states, final_outputs)
    bhi = _bid_heterogeneity_index(signatures)
    pbdc = _profile_behavior_distance_correlation(profiles, signatures)
    ida = _identifiability_accuracy(action_by_park)
    return BehaviorFidelity(rpa=rpa, bhi=bhi, pbdc=pbdc, ida=ida)
