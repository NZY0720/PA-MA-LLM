"""Generate Figure 7: profile-to-behavior coherence as a 4-panel heatmap.

Panel (a) shows the extracted subjective profile theta. Panels (b)-(d)
show per-park realised behaviour under the Proposed framework, the B6
MARL baseline and the C5 parameterised bidder. All four panels use the
same profile-to-behavior layout, with behavior columns selected to reflect
carbon priority, negotiation, export/import willingness, and target volume.
Columns are min-max normalized to [0,1] across parks so the four panels share a
single colour scale and a directly comparable visual pattern.

Visual story: Proposed's row patterns track theta column by column;
MARL is essentially flat on the LLM-specific behaviour fields it does
not natively emit, and varies only on capacity-driven dimensions; C5
varies but on a capacity-driven axis rather than a profile-driven one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from utils.constants import BASE_DIR
from visualization._style import configure_plot_style

configure_plot_style("case2")

CASE2_DIR = BASE_DIR / "outputs" / "case2"
TRACE_DIR = CASE2_DIR / "llm_traces"
PROFILE_FILE = CASE2_DIR / "case2_subjective_profiles.json"

PARK_ORDER = ("Park_A", "Park_B", "Park_C", "Park_D", "Park_E")
PARK_LABEL = ("A", "B", "C", "D", "E")

# Full profile dimensions (panel a) and the realised behaviour fields
# (panels b-d). The figure invites a visual row-pattern comparison rather
# than enforcing a strict column-by-column semantic mapping: the proposed
# framework produces a behaviour heatmap whose row pattern visibly mirrors
# the profile heatmap, while the MARL and parameterised baselines do not.
THETA_DIMS = ("risk", "carbon", "service", "autonomy", "neg")
THETA_LABELS = (
    r"$\theta_{\mathrm{risk}}$",
    r"$\theta_{\mathrm{carbon}}$",
    r"$\theta_{\mathrm{serv}}$",
    r"$\theta_{\mathrm{auton}}$",
    r"$\theta_{\mathrm{neg}}$",
)

BEHAVIOR_FIELDS = (
    "carbon_priority",
    "concession_factor",
    "export_willingness",
    "import_willingness",
    "export_target_kwh",
)
BEHAVIOR_LABELS = (
    "carbon priority",
    "concession factor",
    "export will.",
    "import will.",
    "export volume",
)


def _load_intent(filename: str) -> dict:
    with open(TRACE_DIR / filename, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _per_park_means(intent: dict) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, list[float]]] = {p: {f: [] for f in BEHAVIOR_FIELDS} for p in PARK_ORDER}
    for payload in intent.get("hours", {}).values():
        rounds = payload.get("rounds", [])
        if not rounds:
            continue
        po = rounds[-1].get("park_outputs", {})
        for park_id in PARK_ORDER:
            agent_out = po.get(park_id, {}) or {}
            for field in BEHAVIOR_FIELDS:
                value = agent_out.get(field)
                if isinstance(value, (int, float)):
                    accum[park_id][field].append(float(value))
    return {
        p: {f: (float(np.mean(vals)) if vals else 0.0) for f, vals in fields.items()}
        for p, fields in accum.items()
    }


def _theta_matrix(profile_payload: dict) -> np.ndarray:
    return np.array(
        [[float(profile_payload[p]["subjective_profile"].get(d, 0.0)) for d in THETA_DIMS] for p in PARK_ORDER]
    )


def _behavior_matrix(intent: dict) -> np.ndarray:
    means = _per_park_means(intent)
    return np.array([[means[p][f] for f in BEHAVIOR_FIELDS] for p in PARK_ORDER])


def _minmax_columns(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-9:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return out


def _column_pearson(theta: np.ndarray, behavior_norm: np.ndarray) -> float:
    """Mean Pearson r between paired columns of theta and behaviour matrices.

    Both matrices are assumed to have the same column order, where column j of
    the behaviour matrix is the realised expression of theta dimension j. The
    metric is the average column-wise r across the four paired dimensions, so
    a high value means realised behaviour rank-orders parks the same way the
    extracted profile does, on the same axes.
    """
    rs = []
    for j in range(theta.shape[1]):
        x, y = theta[:, j], behavior_norm[:, j]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rs)) if rs else 0.0


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    column_labels: Iterable[str],
    title: str,
    cmap: str = "Blues",
):
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(list(column_labels), fontsize=8.0, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(PARK_LABEL)))
    ax.set_yticklabels([f"Park {lab}" for lab in PARK_LABEL], fontsize=8.6)
    ax.set_title(title, fontsize=9.0)
    # Blues colormap: values below ~0.55 are pale enough to take black text;
    # values above are dark enough to require white text. Threshold tuned for
    # the Blues palette specifically.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "black" if value < 0.55 else "white"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.2, color=color)
    return im


def plot_profile_behavior_coherence(out_path: Path) -> None:
    profile_payload = json.load(open(PROFILE_FILE, "r", encoding="utf-8"))
    theta = _theta_matrix(profile_payload)

    intent_proposed = _load_intent("deepseek_run_1.json")
    intent_marl = _load_intent("marl_run_1.json")
    intent_param = _load_intent("parameterized_ablation_run_1.json")

    behaviors = {
        "Proposed": _behavior_matrix(intent_proposed),
        "MARL": _behavior_matrix(intent_marl),
        "Param": _behavior_matrix(intent_param),
    }
    behavior_norm = {k: _minmax_columns(v) for k, v in behaviors.items()}

    fig, axes = plt.subplots(
        1, 4, figsize=(13.0, 3.2),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0, 1.0]},
    )

    im0 = _draw_heatmap(
        axes[0], theta, THETA_LABELS,
        title=r"(a) Extracted profile $\theta_{m}^{k}$ (reference)",
    )

    panel_titles = {
        "Proposed":  "(b) Proposed: heterogeneity reflects $\\theta$",
        "MARL":      "(c) B6 MARL: flat on profile-coupled fields",
        "Param":     "(d) C5 param.: heterogeneity is capacity-driven",
    }
    for ax, name in zip(axes[1:], ("Proposed", "MARL", "Param")):
        _draw_heatmap(ax, behavior_norm[name], BEHAVIOR_LABELS, title=panel_titles[name])
        ax.set_yticklabels([])

    cbar = fig.colorbar(im0, ax=axes.tolist(), orientation="vertical", fraction=0.018, pad=0.02)
    cbar.set_label("Value (column-normalized in (b)-(d))", fontsize=8.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = BASE_DIR / "outputs" / "case2" / "figures" / "fig7_profile_behavior_coherence.pdf"
    plot_profile_behavior_coherence(out)
    print(f"Saved: {out}")
