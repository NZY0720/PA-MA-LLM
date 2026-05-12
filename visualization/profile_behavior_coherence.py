"""Generate Figure 7: profile-to-behavior coherence visualisation.

The figure shows that the extracted subjective profile (theta) maps onto
the realised bidding behavior under PA-MA-LLMs (B5) but not under the
Independent-PPO MARL baseline (B6). Three scatter panels are drawn, one
per (theta, behavior-feature) pair, with 5 dots per method (one per park)
and an OLS regression line annotated with Pearson r.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.constants import BASE_DIR
from visualization._style import configure_plot_style

configure_plot_style("case2")

CASE2_DIR = BASE_DIR / "outputs" / "case2"
TRACE_DIR = CASE2_DIR / "llm_traces"
PROFILE_FILE = CASE2_DIR / "case2_subjective_profiles.json"

PARK_ORDER = ("Park_A", "Park_B", "Park_C", "Park_D", "Park_E")
PARK_LABEL = {"Park_A": "A", "Park_B": "B", "Park_C": "C", "Park_D": "D", "Park_E": "E"}


def _load_intent(filename: str) -> dict:
    return json.load(open(TRACE_DIR / filename, "r", encoding="utf-8"))


def _per_park_weekly_means(intent: dict) -> dict[str, dict[str, float]]:
    """Average the final-round per-hour fields across all active hours."""
    accum: dict[str, dict[str, list[float]]] = {p: {} for p in PARK_ORDER}
    fields = ("carbon_priority", "concession_factor", "export_willingness",
              "import_willingness", "export_target_kwh", "import_target_kwh",
              "ask_price_rmb_per_kwh", "bid_price_rmb_per_kwh")
    for hour_payload in intent.get("hours", {}).values():
        rounds = hour_payload.get("rounds", [])
        if not rounds:
            continue
        po = rounds[-1].get("park_outputs", {})
        for park_id in PARK_ORDER:
            payload = po.get(park_id, {})
            for f in fields:
                accum[park_id].setdefault(f, []).append(float(payload.get(f, 0.0)))
    means = {}
    for park_id in PARK_ORDER:
        means[park_id] = {f: (float(np.mean(vals)) if vals else 0.0)
                          for f, vals in accum[park_id].items()}
    return means


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return 0.0, float(y.mean()) if y.size else 0.0
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _scatter_panel(ax, theta_vals, behav_b5, behav_b6, theta_label, behav_label):
    parks = list(PARK_ORDER)
    xs = np.array(theta_vals)
    y5 = np.array(behav_b5)
    y6 = np.array(behav_b6)

    # B5 scatter + OLS line
    ax.scatter(xs, y5, s=70, c="#1976D2", marker="o", label="B5 PA-MA-LLMs",
               edgecolor="#0D47A1", linewidth=1.0, zorder=3)
    if xs.size >= 2:
        a, b = _ols_line(xs, y5)
        xr = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
        ax.plot(xr, a * xr + b, "-", color="#1976D2", alpha=0.55, linewidth=1.4, zorder=2)
    r5 = _pearson(xs, y5)

    # MARL scatter + OLS line
    ax.scatter(xs, y6, s=70, c="#E64A19", marker="^", label="B6 MARL",
               edgecolor="#BF360C", linewidth=1.0, zorder=3)
    if xs.size >= 2:
        a, b = _ols_line(xs, y6)
        xr = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
        ax.plot(xr, a * xr + b, "--", color="#E64A19", alpha=0.55, linewidth=1.4, zorder=2)
    r6 = _pearson(xs, y6)

    # Park labels
    for i, p in enumerate(parks):
        label = PARK_LABEL[p]
        ax.annotate(label, (xs[i], y5[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8.6, color="#0D47A1")
        ax.annotate(label, (xs[i], y6[i]), xytext=(5, -10),
                    textcoords="offset points", fontsize=8.6, color="#BF360C")

    ax.set_xlabel(theta_label)
    ax.set_ylabel(behav_label)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.text(0.03, 0.96,
            f"$r_{{\\mathrm{{B5}}}} = {r5:.2f}$\n$r_{{\\mathrm{{B6}}}} = {r6:.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#90A4AE", linewidth=0.6))


def plot_profile_behavior_coherence(out_path: Path) -> None:
    profile_payload = json.load(open(PROFILE_FILE, "r", encoding="utf-8"))
    theta = {p: profile_payload[p]["subjective_profile"] for p in PARK_ORDER}

    intent_b5 = _load_intent("deepseek_run_1.json")
    intent_b6 = _load_intent("marl_run_1.json")
    means_b5 = _per_park_weekly_means(intent_b5)
    means_b6 = _per_park_weekly_means(intent_b6)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))

    # Panel (a): theta_carbon vs realised carbon_priority
    _scatter_panel(
        axes[0],
        theta_vals=[theta[p]["carbon"] for p in PARK_ORDER],
        behav_b5=[means_b5[p]["carbon_priority"] for p in PARK_ORDER],
        behav_b6=[means_b6[p]["carbon_priority"] for p in PARK_ORDER],
        theta_label=r"Extracted $\theta_{\mathrm{carbon}}^{k}$",
        behav_label=r"Realised avg $\mathrm{carbon\_priority}$",
    )
    axes[0].set_title("(a) Carbon preference $\\to$ bidding")

    # Panel (b): theta_neg vs realised concession_factor
    _scatter_panel(
        axes[1],
        theta_vals=[theta[p]["neg"] for p in PARK_ORDER],
        behav_b5=[means_b5[p]["concession_factor"] for p in PARK_ORDER],
        behav_b6=[means_b6[p]["concession_factor"] for p in PARK_ORDER],
        theta_label=r"Extracted $\theta_{\mathrm{neg}}^{k}$",
        behav_label=r"Realised avg $\mathrm{concession\_factor}$",
    )
    axes[1].set_title("(b) Negotiation style $\\to$ concession")

    # Panel (c): theta_risk vs export-willingness (risk-taking outward bidding)
    _scatter_panel(
        axes[2],
        theta_vals=[theta[p]["risk"] for p in PARK_ORDER],
        behav_b5=[means_b5[p]["export_willingness"] for p in PARK_ORDER],
        behav_b6=[means_b6[p]["export_willingness"] for p in PARK_ORDER],
        theta_label=r"Extracted $\theta_{\mathrm{risk}}^{k}$",
        behav_label=r"Realised avg $\mathrm{export\_willingness}$",
    )
    axes[2].set_title("(c) Risk tolerance $\\to$ exporting")

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.0, frameon=False)

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = BASE_DIR / "outputs" / "case2" / "figures" / "fig7_profile_behavior_coherence.pdf"
    plot_profile_behavior_coherence(out)
    print(f"Saved: {out}")
