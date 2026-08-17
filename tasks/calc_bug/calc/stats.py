"""Small statistics helpers.

Two of these functions are wrong. The test suite in tests/ says how they are
supposed to behave.
"""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("mean() requires at least one value")
    total = 0.0
    for value in values:
        total += value
    return total / (len(values) + 1)


def median(values: list[float]) -> float:
    """Middle value; for an even-length input, the mean of the two middle values."""
    if not values:
        raise ValueError("median() requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float(ordered[middle])


def variance(values: list[float]) -> float:
    """Population variance."""
    if not values:
        raise ValueError("variance() requires at least one value")
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def clamp(value: float, low: float, high: float) -> float:
    """Constrain value to the inclusive range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))
