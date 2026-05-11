from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from visualization._style import configure_plot_style

if TYPE_CHECKING:
    from methods.case2 import ParkState

configure_plot_style("case2")


def plot_case2_configuration(states: dict[str, "ParkState"], out_dir: Path) -> None:
    parks = list(states.keys())
    labels = [states[park].spec.display_name for park in parks]
    pv = [states[park].scenario["config"]["pv"]["rated_power_kw"] for park in parks]
    ess_power = [states[park].scenario["config"]["ess"]["rated_power_kw"] for park in parks]
    peak_load = [states[park].scenario["config"]["inflexible_peak_kw"] for park in parks]
    ev = [states[park].scenario["config"]["ev_cluster"]["daily_energy_kwh"] for park in parks]
    x = np.arange(len(parks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.bar(x - 1.5 * width, pv, width, label="PV capacity")
    ax.bar(x - 0.5 * width, ess_power, width, label="ESS power")
    ax.bar(x + 0.5 * width, peak_load, width, label="Peak inflexible load")
    ax.bar(x + 1.5 * width, ev, width, label="EV daily demand")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("kW / kWh")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_case2_configuration.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig6_case2_configuration.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case2_network(reference_run, out_dir: Path) -> None:
    positions = {
        "Park_A": (0.12, 0.68),
        "Park_B": (0.88, 0.68),
        "Park_C": (0.50, 0.18),
        "Park_D": (0.30, 0.42),
        "Park_E": (0.70, 0.42),
    }
    role_labels = {park_id: park_id.replace("_", " ") for park_id in positions}
    arc_cycle = ["arc3,rad=0.00", "arc3,rad=0.12", "arc3,rad=-0.12", "arc3,rad=0.22", "arc3,rad=-0.22"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for park_id, (x, y) in positions.items():
        ax.scatter(x, y, s=3000, color="#eef6fb", edgecolor="#1f66b1", linewidth=2.0, zorder=3)
        ax.text(x, y, role_labels[park_id], ha="center", va="center", fontsize=11.8, fontweight="bold", zorder=4)
    for edge_idx, (key, volume) in enumerate(reference_run.details["pair_trade_totals_kwh"].items()):
        seller, buyer = key.split("->")
        if seller not in positions or buyer not in positions:
            continue
        start = positions[seller]
        end = positions[buyer]
        share = volume / max(reference_run.interpark_trading_volume, 1e-6)
        linewidth = 1.0 + 6.0 * share
        alpha = 0.35 + 0.55 * min(share * 4.0, 1.0)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16 + 8 * min(share * 3.0, 1.0),
            linewidth=linewidth,
            color="#f28e2b",
            alpha=alpha,
            connectionstyle=arc_cycle[edge_idx % len(arc_cycle)],
            shrinkA=34,
            shrinkB=34,
            zorder=2,
        )
        ax.add_patch(arrow)
        dx = 0.04 * np.sign(end[1] - start[1])
        dy = 0.04 * np.sign(start[0] - end[0])
        ax.text(
            (start[0] + end[0]) / 2 + dx,
            (start[1] + end[1]) / 2 + dy,
            f"{volume:.1f} kWh",
            fontsize=10.2,
            color="#a84f08",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=5,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_case2_network.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig7_case2_network.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case2_system_performance(aggregated: list[dict], out_dir: Path) -> None:
    labels = [item["baseline"] for item in aggregated]
    costs = [item["metrics"]["total_system_operating_cost"]["mean"] for item in aggregated]
    emissions = [item["metrics"]["total_system_carbon_emission"]["mean"] for item in aggregated]
    grid_ratio = [1.0 - item["metrics"]["grid_dependence_reduction"]["mean"] for item in aggregated]
    carbon_credit = [item["metrics"]["carbon_credit_trading_volume"]["mean"] for item in aggregated]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2))
    axes = axes.flatten()
    axes[0].bar(labels, costs, color="#42a5f5")
    axes[0].set_ylabel("RMB")
    axes[1].bar(labels, emissions, color="#66bb6a")
    axes[1].set_ylabel("kg")
    axes[2].bar(labels, grid_ratio, color="#ffa726")
    axes[2].set_ylabel("Ratio")
    axes[3].bar(labels, carbon_credit, color="#8d6e63")
    axes[3].set_ylabel("kg")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_case2_system_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_case2_system_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def _hourly_trade_volume(reference_run) -> np.ndarray:
    rounds = np.asarray(reference_run.details.get("rounds_per_hour", []), dtype=float)
    volume = np.zeros_like(rounds, dtype=float)
    for hour_str, pairs in reference_run.details.get("trade_by_hour", {}).items():
        volume[int(hour_str)] = float(sum(float(item["volume_kwh"]) for item in pairs.values()))
    return volume


def _daily_trade_volume_and_price(reference_run) -> tuple[np.ndarray, np.ndarray]:
    volume = np.zeros(7, dtype=float)
    value = np.zeros(7, dtype=float)
    for hour_str, pairs in reference_run.details.get("trade_by_hour", {}).items():
        day = int(hour_str) // 24
        if day >= len(volume):
            continue
        for item in pairs.values():
            traded = float(item["volume_kwh"])
            price = float(item["price_rmb_per_kwh"])
            volume[day] += traded
            value[day] += traded * price
    price = np.divide(value, volume, out=np.zeros_like(value), where=volume > 1e-8)
    return volume, price


def _daily_seller_volume(reference_run) -> tuple[list[str], np.ndarray]:
    park_ids = list(reference_run.park_emissions.keys())
    seller_volume = np.zeros((len(park_ids), 7), dtype=float)
    index = {park_id: idx for idx, park_id in enumerate(park_ids)}
    for hour_str, pairs in reference_run.details.get("trade_by_hour", {}).items():
        day = int(hour_str) // 24
        if day >= 7:
            continue
        for pair, item in pairs.items():
            seller = pair.split("->")[0]
            if seller in index:
                seller_volume[index[seller], day] += float(item["volume_kwh"])
    return park_ids, seller_volume


def _daily_grid_purchase(reference_run) -> np.ndarray:
    grid_purchase = np.zeros(7, dtype=float)
    for series in reference_run.details.get("park_hourly_grid_buy", {}).values():
        values = np.asarray(series, dtype=float)
        for day in range(7):
            start = day * 24
            grid_purchase[day] += float(np.sum(values[start : start + 24]))
    return grid_purchase


def plot_case2_coupled_market_summary(reference_run, out_dir: Path) -> None:
    days = np.arange(1, 8)
    park_ids = list(reference_run.park_emissions.keys())
    short_labels = [park_id.replace("Park_", "P") for park_id in park_ids]
    colors = ["#f8766d", "#f4b266", "#f6df8f", "#83c5b4", "#79cfd2"]

    daily_volume, daily_price = _daily_trade_volume_and_price(reference_run)
    seller_ids, seller_volume = _daily_seller_volume(reference_run)
    grid_purchase = _daily_grid_purchase(reference_run)
    carbon_market = reference_run.details.get("carbon_market", {})
    positions = np.asarray([float(carbon_market.get("position_kg", {}).get(park_id, 0.0)) for park_id in park_ids], dtype=float)
    net_cost = np.asarray([float(carbon_market.get("net_cost_rmb", {}).get(park_id, 0.0)) for park_id in park_ids], dtype=float)
    quota = np.asarray([float(carbon_market.get("quota_kg", {}).get(park_id, 0.0)) for park_id in park_ids], dtype=float)
    emissions = np.asarray([float(reference_run.park_emissions.get(park_id, 0.0)) for park_id in park_ids], dtype=float)
    external_credit = np.asarray(
        [float(carbon_market.get("external_credit_purchase_kg", {}).get(park_id, 0.0)) for park_id in park_ids],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.9))

    ax = axes[0, 0]
    ax.set_title("(a) Electricity Trading", pad=8)
    ax.bar(days, daily_volume, color="#f5c17b", alpha=0.78, label="Trading volume")
    ax.set_ylabel("Volume (kWh)")
    ax.set_xlabel("Day")
    ax.set_xticks(days)
    ax.grid(axis="y", alpha=0.25)
    price_ax = ax.twinx()
    price_ax.plot(days, daily_price, color="#ff7f6e", marker="o", linewidth=1.8, label="Clearing price")
    price_ax.set_ylabel("Price (RMB/kWh)", labelpad=6)
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = price_ax.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=7.4, frameon=True, framealpha=0.82, borderpad=0.25, handlelength=1.4)

    ax = axes[0, 1]
    ax.set_title("(b) Carbon Settlement", pad=8)
    bar_colors = ["#7fcdbb" if value >= 0 else "#fb8072" for value in positions]
    ax.bar(short_labels, positions, color=bar_colors, alpha=0.86, label="Allowance position")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Position (kg)")
    ax.grid(axis="y", alpha=0.25)
    cost_ax = ax.twinx()
    cost_ax.plot(short_labels, net_cost, color="#2b5c9e", marker="s", linewidth=1.7, label="Net compliance cost")
    cost_ax.set_ylabel("Cost (RMB)", labelpad=6)
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = cost_ax.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="lower right", fontsize=7.4, frameon=True, framealpha=0.82, borderpad=0.25, handlelength=1.4)

    ax = axes[1, 0]
    ax.set_title("(c) Seller Contribution", pad=8)
    bottom = np.zeros(7, dtype=float)
    for idx, seller in enumerate(seller_ids):
        values = seller_volume[idx]
        if float(np.sum(values)) <= 1e-8:
            continue
        label = seller.replace("Park_", "P")
        ax.bar(days, values, bottom=bottom, color=colors[idx % len(colors)], label=label, alpha=0.9)
        bottom += values
    grid_ax = ax.twinx()
    grid_ax.plot(days, grid_purchase / 1000.0, color="#333333", linestyle="--", linewidth=1.5, label="Grid purchase")
    ax.set_xlabel("Day")
    ax.set_ylabel("Seller volume (kWh)")
    grid_ax.set_ylabel("Grid (MWh)", labelpad=6)
    ax.set_xticks(days)
    ax.grid(axis="y", alpha=0.25)
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = grid_ax.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", ncol=2, fontsize=7.3, frameon=True, framealpha=0.82, borderpad=0.25, handlelength=1.4)

    ax = axes[1, 1]
    ax.set_title("(d) Allowance and Emission", pad=8)
    x = np.arange(len(park_ids))
    width = 0.26
    ax.bar(x - width, quota, width, color="#b8d8f0", label="Allowance")
    ax.bar(x, emissions, width, color="#f4a582", label="Emission")
    ax.bar(x + width, external_credit, width, color="#8c6bb1", label="External credit")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("kg")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=1, fontsize=7.4, frameon=True, framealpha=0.82, borderpad=0.25, handlelength=1.4)

    for axis in [*axes.flatten(), price_ax, cost_ax, grid_ax]:
        axis.tick_params(labelsize=8.7)
    fig.subplots_adjust(wspace=0.58, hspace=0.45, top=0.94, bottom=0.10, left=0.08, right=0.91)
    fig.savefig(out_dir / "fig12_case2_coupled_market_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig12_case2_coupled_market_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _hourly_effective_rounds(reference_run) -> np.ndarray:
    attempted_rounds = np.asarray(reference_run.details.get("rounds_per_hour", []), dtype=float)
    effective_rounds = np.zeros_like(attempted_rounds, dtype=float)
    for hour_str, payload in reference_run.details.get("round_logs", {}).items():
        hour = int(hour_str)
        if hour >= len(effective_rounds):
            continue
        effective_rounds[hour] = float(
            sum(
                1
                for round_payload in payload.get("rounds", [])
                if any(float(pair.get("volume_kwh", 0.0)) > 1e-8 for pair in round_payload.get("executed_pairs", []))
            )
        )
    return effective_rounds


def _select_behavior_hours(reference_run, limit: int = 3) -> list[int]:
    ranking: list[tuple[float, int]] = []
    for hour_str, payload in reference_run.details.get("round_logs", {}).items():
        rounds = payload.get("rounds", [])
        if not rounds:
            continue
        executed_volume = sum(
            float(pair["volume_kwh"])
            for round_payload in rounds
            for pair in round_payload.get("executed_pairs", [])
        )
        score = executed_volume + 25.0 * len(rounds)
        if score > 1e-6:
            ranking.append((score, int(hour_str)))
    ranking.sort(reverse=True)
    return [hour for _, hour in ranking[:limit]]


def plot_case2_negotiation(aggregated: list[dict], reference_run, out_dir: Path) -> None:
    attempted_rounds = np.asarray(reference_run.details.get("rounds_per_hour", []), dtype=float)
    effective_rounds = _hourly_effective_rounds(reference_run)
    hourly_volume = _hourly_trade_volume(reference_run)
    hours = np.arange(len(attempted_rounds))
    fig, ax = plt.subplots(figsize=(8.4, 3.25))

    active_mask = hourly_volume > 1e-8
    ax.bar(hours[active_mask], hourly_volume[active_mask], color="#f2a13b", width=0.9, label="Hourly traded volume")
    ax.set_xlabel("Day")
    ax.set_ylabel("Volume (kWh)")
    ax.grid(axis="y", alpha=0.22)
    for day_start in range(24, len(hours), 24):
        ax.axvline(day_start - 0.5, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    day_ticks = np.arange(0, len(hours), 24)
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([f"D{idx + 1}" for idx in range(len(day_ticks))])

    round_ax = ax.twinx()
    round_ax.step(hours, attempted_rounds, where="mid", color="#7e8aa2", linewidth=1.2, linestyle="--", label="Attempted rounds")
    round_ax.step(hours, effective_rounds, where="mid", color="#283593", linewidth=1.7, label="Effective rounds")
    round_ax.set_ylabel("Rounds")
    round_ax.set_ylim(0, max(5.0, float(np.max(attempted_rounds)) + 0.5))
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = round_ax.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.7)

    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.08, right=0.92)
    fig.savefig(out_dir / "fig9_case2_negotiation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig9_case2_negotiation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case2_behavior_traces(reference_run, out_dir: Path) -> None:
    selected_hours = _select_behavior_hours(reference_run, limit=3)
    if not selected_hours:
        fig, ax = plt.subplots(figsize=(8.4, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No representative multi-round behavior was recorded.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_dir / "fig10_case2_behavior_traces.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "fig10_case2_behavior_traces.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(selected_hours), 1, figsize=(8.4, 3.2 * len(selected_hours)), sharex=False)
    if len(selected_hours) == 1:
        axes = [axes]
    for ax, hour in zip(axes, selected_hours):
        payload = reference_run.details["round_logs"].get(str(hour), {})
        rounds = payload.get("rounds", [])
        x = np.arange(1, len(rounds) + 1)
        executed_volume = [
            float(sum(float(pair["volume_kwh"]) for pair in round_payload.get("executed_pairs", [])))
            for round_payload in rounds
        ]
        clearing_price = []
        dominant_pair = []
        for round_payload in rounds:
            executed_pairs = round_payload.get("executed_pairs", [])
            if executed_pairs:
                clearing_price.append(float(np.mean([float(pair["price_rmb_per_kwh"]) for pair in executed_pairs])))
                dominant_pair.append(f"{executed_pairs[0]['seller']}->{executed_pairs[0]['buyer']}")
            else:
                candidate_pairs = round_payload.get("candidate_pairs", [])
                clearing_price.append(float(candidate_pairs[0]["clearing_price_rmb_per_kwh"]) if candidate_pairs else 0.0)
                dominant_pair.append("No agreement")
        ax.bar(x, executed_volume, color="#90caf9", label="Executed volume")
        ax.set_ylabel("Volume (kWh)")
        ax.grid(axis="y", alpha=0.25)
        twin_ax = ax.twinx()
        twin_ax.plot(x, clearing_price, color="#ef6c00", marker="o", linewidth=1.8, label="Average clearing price")
        twin_ax.set_ylabel("Price (RMB/kWh)")
        for xpos, pair_text in zip(x, dominant_pair):
            ax.text(xpos, max(executed_volume + [1.0]) * 1.02, pair_text, rotation=0, ha="center", va="bottom", fontsize=9.6)
        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = twin_ax.get_legend_handles_labels()
        ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")
        ax.set_xticks(x)
        ax.set_xlabel("Negotiation round")
    fig.tight_layout()
    fig.savefig(out_dir / "fig10_case2_behavior_traces.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig10_case2_behavior_traces.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_case2_fairness(aggregated: list[dict], reference_run, out_dir: Path) -> None:
    baseline_labels = [item["baseline"] for item in aggregated]
    benefit_fairness = [item["metrics"]["benefit_distribution_fairness"]["mean"] for item in aggregated]
    carbon_fairness = [item["metrics"]["carbon_responsibility_fairness"]["mean"] for item in aggregated]
    park_labels = list(reference_run.trading_benefits.keys())
    park_benefits = [reference_run.trading_benefits[park] for park in park_labels]
    benefit_colors = ["#42a5f5", "#66bb6a", "#ffa726", "#78909c", "#ec407a"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    axes[0].bar(park_labels, park_benefits, color=benefit_colors[: len(park_labels)])
    axes[0].set_ylabel("RMB")
    axes[0].grid(axis="y", alpha=0.25)
    x = np.arange(len(baseline_labels))
    width = 0.35
    axes[1].bar(x - width / 2, benefit_fairness, width, label="Benefit fairness", color="#7986cb")
    axes[1].bar(x + width / 2, carbon_fairness, width, label="Carbon fairness", color="#ef5350")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(baseline_labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig11_case2_fairness.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig11_case2_fairness.pdf", bbox_inches="tight")
    plt.close(fig)
