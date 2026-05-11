from __future__ import annotations

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_np(values: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def jains_index(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    denominator = array.size * np.square(array).sum()
    if denominator <= 1e-12:
        return 0.0
    return float((array.sum() ** 2) / denominator)


def series_payload(values: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in values.tolist()]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")
