from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils.constants import FIGURES_OUTPUT_DIR, SCENARIO_PATH
from utils.io_utils import load_json
from visualization._style import configure_plot_style

configure_plot_style("structure")


def _add_box(ax, xy, text, width=2.4, height=0.9, fc="#f5f7fa", ec="#455a64", fontsize=11):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x + width / 2, y + height / 2)


def _add_arrow(ax, start, end, text="", color="#546e7a"):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color)
    ax.add_patch(arrow)
    if text:
        tx = (start[0] + end[0]) / 2
        ty = (start[1] + end[1]) / 2
        ax.text(tx, ty + 0.12, text, fontsize=9.8, color=color, ha="center")


def draw_park_structure(scenario: dict, out_dir: Path) -> None:
    config = scenario["config"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(7, 9.6, "Weekly Low-Carbon Park Structure", ha="center", va="center", fontsize=12.6, fontweight="bold")
    market = _add_box(ax, (0.8, 7.8), "Market Environment\nPrice + Carbon + Weather", width=3.2, fc="#fff8e1")
    grid = _add_box(ax, (10.2, 7.8), f"External Grid\nTie-line: {config['tie_line_limit_kw']:.0f} kW", width=3.0, fc="#ede7f6")
    operator = _add_box(ax, (5.3, 6.2), "Park Operator Agent", width=3.4, fc="#e8f5e9")
    pv = _add_box(ax, (1.0, 3.9), f"PV\n{config['pv']['rated_power_kw']:.0f} kW", width=2.2, fc="#fff3e0")
    ess = _add_box(ax, (3.7, 3.9), f"ESS\n{config['ess']['rated_power_kw']:.0f} kW / {config['ess']['energy_capacity_kwh']:.0f} kWh", width=3.0, fc="#e3f2fd", fontsize=10)
    inflexible = _add_box(ax, (7.2, 3.9), f"Inflexible Load\nPeak {config['inflexible_peak_kw']:.0f} kW", width=2.7, fc="#fce4ec")
    hvac = _add_box(ax, (10.3, 3.9), "Flexible Load\nHVAC", width=2.2, fc="#f1f8e9")
    ev = _add_box(ax, (5.2, 2.1), f"EV Cluster\n{config['ev_cluster']['slots']} slots / {config['ev_cluster']['max_charging_power_kw']:.0f} kW", width=3.7, fc="#e0f7fa", fontsize=10)
    _add_arrow(ax, market, operator, "7x24 signals")
    _add_arrow(ax, operator, grid, "Electricity-carbon settlement")
    _add_arrow(ax, operator, pv, "dispatch")
    _add_arrow(ax, operator, ess, "charge/discharge")
    _add_arrow(ax, operator, inflexible, "supply")
    _add_arrow(ax, operator, hvac, "thermal flexibility")
    _add_arrow(ax, operator, ev, "charging schedule")
    fig.tight_layout()
    fig.savefig(out_dir / "park_structure.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "park_structure.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_experiment_profiles(scenario: dict, out_dir: Path) -> None:
    profiles = scenario["profiles"]
    hours = profiles["hours"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(hours, profiles["buy_price_rmb_per_kwh"], label="Buy Price (RMB/kWh)", linewidth=2)
    axes[0].plot(hours, profiles["grid_carbon_intensity_kg_per_kwh"], label="Carbon Intensity (kg/kWh)", linewidth=2)
    axes[0].set_ylabel("Signal Value")
    axes[0].set_title("Weekly Market Signals")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[1].plot(hours, profiles["inflexible_load_kw"], label="Inflexible Load", linewidth=2)
    axes[1].plot(hours, profiles["flexible_loads_kw"]["hvac_load"], label="HVAC Load", linewidth=2)
    axes[1].plot(hours, profiles["flexible_loads_kw"]["service_load"], label="Service Load", linewidth=2)
    axes[1].plot(hours, profiles["pv_available_kw"], label="PV Available", linewidth=2)
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_title("Weekly Main Power Profiles")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    axes[2].bar(hours, profiles["ev_energy_request_kwh"], label="EV Energy Request", alpha=0.7)
    axes[2].set_xlabel("Hour of Week")
    axes[2].set_ylabel("Energy (kWh)")
    axes[2].set_title("Weekly EV Charging Request")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")
    fig.suptitle("Weekly Experiment Scenario (7x24)", fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "experiment_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "experiment_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def render_default_visualizations() -> None:
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = load_json(SCENARIO_PATH)
    draw_park_structure(scenario, FIGURES_OUTPUT_DIR)
    draw_experiment_profiles(scenario, FIGURES_OUTPUT_DIR)
