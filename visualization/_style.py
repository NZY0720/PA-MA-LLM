from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

_BASE_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_SIZE_PRESETS = {
    "case1": {
        "font.size": 9.8,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.4,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.8,
    },
    "case2": {
        "font.size": 9.6,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.3,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.3,
    },
    "structure": {
        "font.size": 11.2,
        "axes.labelsize": 11.4,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.6,
        "ytick.labelsize": 10.6,
        "legend.fontsize": 10.4,
    },
}


def configure_plot_style(preset: str) -> None:
    plt.rcParams.update({**_BASE_RCPARAMS, **_SIZE_PRESETS[preset]})
