"""Analytical access- and search-pattern accounting for PRR/IRR reports."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .xdp import check_probability


def repeated_report_eta(
    repeats: int, permanent_flip: float, instantaneous_flip: float
) -> float:
    """Pure privacy cost of one differing query coordinate over R reports."""

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    a = check_probability(permanent_flip, "permanent_flip")
    b = check_probability(instantaneous_flip, "instantaneous_flip")
    numerator = (1.0 - a) * (1.0 - b) ** repeats + a * b**repeats
    denominator = a * (1.0 - b) ** repeats + (1.0 - a) * b**repeats
    return math.log(numerator / denominator)


def _reuse_split_distributions(
    repeats: int,
    permanent_flip: float,
    instantaneous_flip: float,
    final_clean_bit_differs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grouped sequence laws for reuse and independent-final-PRR worlds."""

    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("search-pattern accounting needs at least two reports")
    a = check_probability(permanent_flip, "permanent_flip")
    b = check_probability(instantaneous_flip, "instantaneous_flip")
    first_count = repeats - 1
    reuse_probabilities = []
    split_probabilities = []
    effective_flip = a + b - 2.0 * a * b

    for ones_first in range(first_count + 1):
        multiplicity = math.comb(first_count, ones_first)
        first_law = multiplicity * (
            (1.0 - a)
            * b**ones_first
            * (1.0 - b) ** (first_count - ones_first)
            + a
            * (1.0 - b) ** ones_first
            * b ** (first_count - ones_first)
        )
        for last_value in (0, 1):
            total_ones = ones_first + last_value
            reuse = multiplicity * (
                (1.0 - a)
                * b**total_ones
                * (1.0 - b) ** (repeats - total_ones)
                + a
                * (1.0 - b) ** total_ones
                * b ** (repeats - total_ones)
            )
            if final_clean_bit_differs:
                probability_last_one = 1.0 - effective_flip
            else:
                probability_last_one = effective_flip
            final_law = (
                probability_last_one
                if last_value == 1
                else 1.0 - probability_last_one
            )
            reuse_probabilities.append(reuse)
            split_probabilities.append(first_law * final_law)

    reuse_array = np.asarray(reuse_probabilities, dtype=np.float64)
    split_array = np.asarray(split_probabilities, dtype=np.float64)
    reuse_array /= float(reuse_array.sum())
    split_array /= float(split_array.sum())
    return reuse_array, split_array


def search_pattern_gamma(
    repeats: int,
    permanent_flip: float,
    instantaneous_flip: float,
    final_clean_bit_differs: bool,
) -> float:
    """Maximum symmetric log-likelihood ratio for one coordinate."""

    reuse, split = _reuse_split_distributions(
        repeats,
        permanent_flip,
        instantaneous_flip,
        final_clean_bit_differs,
    )
    if np.any(reuse <= 0.0) or np.any(split <= 0.0):
        return float("inf")
    return float(np.max(np.abs(np.log(reuse / split))))


def query_privacy_rows(
    differing_coordinates: int,
    total_coordinates: int,
    repeats: Sequence[int],
    permanent_flip: float,
    instantaneous_flip: float,
) -> List[Dict[str, float]]:
    """Return access- and search-pattern pure budgets for one query pair."""

    differing = int(differing_coordinates)
    total = int(total_coordinates)
    if not 0 <= differing <= total:
        raise ValueError("differing_coordinates must lie in [0,total_coordinates]")
    rows = []
    for repeat in repeats:
        repeat = int(repeat)
        eta = repeated_report_eta(repeat, permanent_flip, instantaneous_flip)
        row = {
            "repeats": float(repeat),
            "differing_coordinates": float(differing),
            "total_coordinates": float(total),
            "access_pattern_eta_per_differing_coordinate": eta,
            "access_pattern_pair_xdp_xi": eta * differing,
        }
        if repeat >= 2:
            gamma_equal = search_pattern_gamma(
                repeat,
                permanent_flip,
                instantaneous_flip,
                final_clean_bit_differs=False,
            )
            gamma_different = search_pattern_gamma(
                repeat,
                permanent_flip,
                instantaneous_flip,
                final_clean_bit_differs=True,
            )
            row.update(
                {
                    "search_pattern_gamma_equal": gamma_equal,
                    "search_pattern_gamma_different": gamma_different,
                    "search_pattern_pair_epsilon": (
                        (total - differing) * gamma_equal
                        + differing * gamma_different
                    ),
                }
            )
        rows.append(row)
    return rows

