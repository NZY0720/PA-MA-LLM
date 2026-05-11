from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from methods.common import BaselineRun, base_arrays
from utils.constants import DAY_NAMES, HOURS_PER_DAY
from utils.math_utils import to_np
from visualization._style import configure_plot_style

configure_plot_style("case1")


def _configure_week_axis(ax, horizon: int) -> None:
    tick_positions = np.arange(0, horizon + 1, HOURS_PER_DAY)
    tick_labels = [DAY_NAMES[idx % len(DAY_NAMES)] for idx in range(len(tick_positions))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)


def plot_case1_profiles(scenario: dict, out_dir: Path) -> None:
    arrays = base_arrays(scenario)
    total_load = arrays["inflexible"] + arrays["hvac_baseline"] + arrays["service_baseline"] + arrays["ev_request"]
    hours = arrays["hours"]

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.0), sharex=True)
    axes[0, 0].plot(hours, total_load, linewidth=1.8, color="#1565c0")
    axes[0, 0].set_title("Weekly Load Profile")
    axes[0, 0].set_ylabel("kW")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(hours, arrays["pv"], linewidth=1.8, color="#ef6c00")
    axes[0, 1].set_title("Weekly PV Profile")
    axes[0, 1].set_ylabel("kW")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(hours, arrays["buy_price"], linewidth=1.8, color="#2e7d32")
    axes[1, 0].set_title("Electricity Price")
    axes[1, 0].set_ylabel("RMB/kWh")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(hours, arrays["carbon_intensity"], linewidth=1.8, color="#6a1b9a")
    axes[1, 1].set_title("Grid Carbon Intensity")
    axes[1, 1].set_ylabel("kg/kWh")
    axes[1, 1].grid(alpha=0.25)
    for ax in axes.flat:
        _configure_week_axis(ax, len(hours) - 1)
    fig.suptitle("Fig. 1. Weekly load, PV, electricity price, and carbon intensity profiles", fontsize=11.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig1_case1_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_case1_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case1_dispatch(reference_run: BaselineRun, scenario: dict, out_dir: Path) -> None:
    arrays = base_arrays(scenario)
    hourly = reference_run.details["hourly"]
    hours = arrays["hours"]
    buy = to_np(hourly["buy_kw"])
    ess = to_np(hourly["ess_kw"])
    hvac = to_np(hourly["hvac_kw"])
    service = to_np(hourly["service_kw"])
    ev = to_np(hourly["ev_kw"])
    total_load = to_np(hourly["total_load_kw"])
    soc = to_np(hourly["soc_kwh"])
    ess_charge = np.maximum(-ess, 0.0)
    ess_discharge = np.maximum(ess, 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True)
    axes[0].stackplot(hours, arrays["pv"], ess_discharge, buy, labels=["PV", "ESS discharge", "Grid purchase"], colors=["#ffb74d", "#64b5f6", "#90a4ae"], alpha=0.85)
    axes[0].plot(hours, total_load + ess_charge, color="black", linewidth=1.8, label="Load + ESS charge")
    axes[0].set_ylabel("Power (kW)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.8)

    axes[1].plot(hours, hvac, label="HVAC load", linewidth=1.6)
    axes[1].plot(hours, service, label="Service load", linewidth=1.6)
    axes[1].plot(hours, ev, label="EV charging", linewidth=1.6)
    soc_ax = axes[1].twinx()
    soc_ax.plot(hours, soc, label="ESS SOC", linewidth=1.5, linestyle="--", color="#d32f2f")
    axes[1].set_ylabel("Load (kW)")
    soc_ax.set_ylabel("ESS SOC (kWh)")
    axes[1].grid(alpha=0.25)
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = soc_ax.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.8)
    _configure_week_axis(axes[1], len(hours) - 1)
    fig.subplots_adjust(hspace=0.42, top=0.90, bottom=0.10, left=0.08, right=0.92)
    fig.savefig(out_dir / "fig2_case1_dispatch.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_case1_dispatch.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case1_performance(aggregated: list[dict], out_dir: Path) -> None:
    labels = [item["baseline"] for item in aggregated]
    costs = [item["metrics"]["total_operating_cost"]["mean"] for item in aggregated]
    emissions = [item["metrics"]["total_carbon_emission"]["mean"] for item in aggregated]
    renewable = [item["metrics"]["renewable_utilization"]["mean"] * 100.0 for item in aggregated]
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.0))
    axes[0].bar(labels, costs, color="#42a5f5")
    axes[0].set_ylabel("RMB")
    axes[1].bar(labels, emissions, color="#66bb6a")
    axes[1].set_ylabel("kg")
    axes[2].bar(labels, renewable, color="#ffa726")
    axes[2].set_ylabel("%")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_case1_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_case1_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case1_carbon(reference_run: BaselineRun, out_dir: Path) -> None:
    carbon = reference_run.details["carbon_responsibility"]
    hours = np.arange(len(carbon["inflexible_load"]))
    fig, ax = plt.subplots(figsize=(8.3, 3.0))
    ax.stackplot(hours, to_np(carbon["inflexible_load"]), to_np(carbon["hvac_load"]), to_np(carbon["service_load"]), to_np(carbon["ev_cluster"]), labels=["Inflexible load", "HVAC", "Service load", "EV cluster"], colors=["#90a4ae", "#42a5f5", "#66bb6a", "#ab47bc"], alpha=0.9)
    _configure_week_axis(ax, len(hours) - 1)
    ax.set_ylabel("Carbon responsibility (kg)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_case1_carbon_allocation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_case1_carbon_allocation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case1_feasibility(aggregated_by_name: dict[str, dict], out_dir: Path) -> None:
    labels = ["LLM-MAS w/o Physics", "PA-MA-LLMs"]
    llm_plain = aggregated_by_name["B4_LLM_MAS_wo_Physics"]["metrics"]
    llm_pi = aggregated_by_name["B5_PA_MA_LLMs"]["metrics"]
    violation = [llm_plain["constraint_violation_rate"]["mean"], llm_pi["constraint_violation_rate"]["mean"]]
    balance = [llm_plain["power_balance_error"]["mean"], llm_pi["power_balance_error"]["mean"]]
    std_cost = [llm_plain["total_operating_cost"]["std"], llm_pi["total_operating_cost"]["std"]]
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.0))
    axes[0].bar(labels, violation, color=["#ef5350", "#42a5f5"])
    axes[0].set_ylabel("Violation rate")
    axes[1].bar(labels, balance, color=["#ef5350", "#42a5f5"])
    axes[1].set_ylabel("kW")
    axes[2].bar(labels, std_cost, color=["#ef5350", "#42a5f5"])
    axes[2].set_ylabel("RMB")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_case1_feasibility.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig5_case1_feasibility.pdf", bbox_inches="tight")
    plt.close(fig)
