"""Linkage-aware empirical XDP accounting for reconstructed multi-shard views.

The fixed linkage methods are executed before this module is called.  For one
challenge pair and one trial, the evaluator records the aggregate number of
clean hash coordinates exposed through *correctly reconstructed* shard
blocks.  Incorrectly reconstructed blocks remain in the end-to-end distance
score, but they do not enter this effective-coordinate count: under the
declared null model they have the same distribution in both neighbouring
worlds and hence have privacy loss zero.

This module turns the empirical effective-coordinate distribution into:

* an exact randomized-response PLD point estimate;
* a high-confidence, high-probability pure-loss limit; and
* an angular-distance-indexed empirical XDP curve.

The PLD point estimate is conditional on the empirical reconstruction
distribution.  The tolerance-limit rows explicitly account for finite sample
size and are the appropriate rows for statements of the form

    Pr[privacy loss > xi] <= beta

with the reported statistical confidence.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .angular_envelope import one_sided_tolerance_limit
from .pld import route_mixture_rr_pld
from .xdp import rr_epsilon_bit


def _integer_distribution(values: Sequence[int]) -> Dict[int, float]:
    samples = np.asarray(values, dtype=np.int64).reshape(-1)
    if samples.size == 0:
        raise ValueError("effective-distance samples cannot be empty")
    if np.any(samples < 0):
        raise ValueError("effective distances cannot be negative")
    unique, counts = np.unique(samples, return_counts=True)
    total = float(samples.size)
    return {
        int(value): float(count) / total
        for value, count in zip(unique.tolist(), counts.tolist())
    }


def _histogram(values: Sequence[int]) -> Dict[str, int]:
    samples = np.asarray(values, dtype=np.int64).reshape(-1)
    unique, counts = np.unique(samples, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique.tolist(), counts.tolist())
    }


def _maximum_two_world_epsilon(
    first_distribution: Mapping[int, float],
    second_distribution: Mapping[int, float],
    flip_probability: float,
    delta: float,
) -> Dict[str, float]:
    first = route_mixture_rr_pld(first_distribution, flip_probability)
    second = route_mixture_rr_pld(second_distribution, flip_probability)
    epsilon0 = float(first.epsilon_for_delta(float(delta)))
    epsilon1 = float(second.epsilon_for_delta(float(delta)))
    return {
        "world0_epsilon": epsilon0,
        "world1_epsilon": epsilon1,
        "symmetric_epsilon": max(epsilon0, epsilon1),
    }


def linkage_aware_xdp(
    effective_distances_world0: Sequence[int],
    effective_distances_world1: Sequence[int],
    flip_probability: float,
    angular_distance: float,
    deltas: Sequence[float],
    tail_probabilities: Sequence[float],
    confidence_beta_per_limit: float,
) -> Dict[str, object]:
    """Account for the hash coordinates exposed by a fixed reconstruction.

    ``effective_distances_world*`` contain one integer per held-out trial.
    Each integer is the sum of the actual shard-specific clean-code Hamming
    distances over correctly reconstructed target blocks in that trial.
    """

    world0 = np.asarray(effective_distances_world0, dtype=np.int64).reshape(-1)
    world1 = np.asarray(effective_distances_world1, dtype=np.int64).reshape(-1)
    if world0.size == 0 or world1.size == 0:
        raise ValueError("both neighbouring worlds require held-out samples")
    if np.any(world0 < 0) or np.any(world1 < 0):
        raise ValueError("effective distances cannot be negative")
    distance = float(angular_distance)
    if not 0.0 < distance <= 1.0:
        raise ValueError("angular_distance must lie in (0,1]")
    if not 0.0 < float(confidence_beta_per_limit) < 1.0:
        raise ValueError("confidence_beta_per_limit must lie in (0,1)")

    distribution0 = _integer_distribution(world0)
    distribution1 = _integer_distribution(world1)
    eta = rr_epsilon_bit(float(flip_probability))

    delta_rows = []
    for delta in deltas:
        row = _maximum_two_world_epsilon(
            distribution0,
            distribution1,
            flip_probability=float(flip_probability),
            delta=float(delta),
        )
        total = float(row["symmetric_epsilon"])
        delta_rows.append(
            {
                "delta": float(delta),
                "xdp_total_budget_xi_point": total,
                "xdp_budget_world0": float(row["world0_epsilon"]),
                "xdp_budget_world1": float(row["world1_epsilon"]),
                "xdp_per_unit_angular_distance_point": total / distance,
            }
        )

    tail_rows = []
    for tail in tail_probabilities:
        limits = {
            0: one_sided_tolerance_limit(
                world0,
                tail_probability=float(tail),
                confidence_beta=float(confidence_beta_per_limit),
            ),
            1: one_sided_tolerance_limit(
                world1,
                tail_probability=float(tail),
                confidence_beta=float(confidence_beta_per_limit),
            ),
        }
        resolved = all(value.sufficient_samples for value in limits.values())
        if resolved:
            distance_limit = int(
                max(
                    round(float(value.upper_limit))
                    for value in limits.values()
                    if value.upper_limit is not None
                )
            )
            xdp_limit: Optional[float] = float(eta * distance_limit)
            coefficient: Optional[float] = xdp_limit / distance
        else:
            distance_limit = None
            xdp_limit = None
            coefficient = None
        tail_rows.append(
            {
                "tail_probability": float(tail),
                "status": "resolved" if resolved else "insufficient_samples",
                "effective_distance_upper_limit": distance_limit,
                "xdp_pure_loss_upper_limit": xdp_limit,
                "xdp_per_unit_angular_distance_upper_limit": coefficient,
                "world0_tolerance": limits[0].to_dict(),
                "world1_tolerance": limits[1].to_dict(),
            }
        )

    combined = np.concatenate([world0, world1])
    return {
        "definition": (
            "linkage-aware empirical XDP obtained from the aggregate actual "
            "fixed-IsoHash distance of correctly reconstructed target blocks"
        ),
        "angular_distance": distance,
        "flip_probability": float(flip_probability),
        "rr_epsilon_per_differing_bit": eta,
        "failed_block_rule": (
            "incorrect or absent blocks remain in the end-to-end reconstructed "
            "score but contribute zero effective coordinates under the common "
            "world-independent null model"
        ),
        "pld_scope": (
            "exact RR PLD conditional on the empirical reconstruction-distance "
            "distribution"
        ),
        "mean_effective_distance": float(combined.mean()),
        "median_effective_distance": float(np.median(combined)),
        "maximum_observed_effective_distance": int(combined.max()),
        "world0_effective_distance_histogram": _histogram(world0),
        "world1_effective_distance_histogram": _histogram(world1),
        "delta_rows": delta_rows,
        "tail_rows": tail_rows,
    }


def reconstructed_score_as_xdp(
    privacy_profile: Mapping[str, object],
    angular_distance: float,
) -> Dict[str, object]:
    """Annotate an end-to-end two-world score profile as pair-specific XDP."""

    distance = float(angular_distance)
    if not 0.0 < distance <= 1.0:
        raise ValueError("angular_distance must lie in (0,1]")
    rows = []
    for source in privacy_profile["delta_rows"]:
        point = source.get("epsilon_point")
        confidence = source.get("epsilon_confidence_upper")
        rows.append(
            {
                "delta": float(source["delta"]),
                "xdp_total_budget_xi_point": point,
                "xdp_total_budget_xi_confidence": confidence,
                "xdp_per_unit_angular_distance_point": (
                    None if point is None else float(point) / distance
                ),
                "xdp_per_unit_angular_distance_confidence": (
                    None
                    if confidence is None
                    else float(confidence) / distance
                ),
                "confidence_status": source["confidence_status"],
            }
        )
    return {
        "definition": (
            "pair-specific empirical XDP of the fixed reconstructed-vector "
            "distance-difference score"
        ),
        "angular_distance": distance,
        "profile_confidence": privacy_profile["profile_confidence"],
        "delta_rows": rows,
    }


def angular_xdp_envelope(
    pair_rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    angular_radii: Sequence[float],
    deltas: Sequence[float],
) -> Dict[str, object]:
    """Aggregate pair-specific results into a tested angular-XDP curve."""

    output = []
    for radius in sorted(set(float(value) for value in angular_radii)):
        if not 0.0 < radius <= 1.0:
            raise ValueError("angular radii must lie in (0,1]")
        eligible = [
            pair
            for pair in pair_rows
            if float(pair["angular_distance"]) <= radius + 1e-12
        ]
        for method in methods:
            for delta_index, delta in enumerate(deltas):
                if not eligible:
                    output.append(
                        {
                            "angular_radius": radius,
                            "method": method,
                            "delta": float(delta),
                            "status": "no_eligible_pairs",
                            "evaluated_pairs": 0,
                            "linkage_xdp_xi_point_envelope": None,
                            "score_xdp_xi_point_envelope": None,
                            "score_xdp_xi_confidence_envelope": None,
                        }
                    )
                    continue
                linkage_values = [
                    float(
                        pair["methods"][method]["linkage_aware_xdp"][
                            "delta_rows"
                        ][delta_index]["xdp_total_budget_xi_point"]
                    )
                    for pair in eligible
                ]
                score_rows = [
                    pair["methods"][method]["reconstructed_vector_xdp"][
                        "delta_rows"
                    ][delta_index]
                    for pair in eligible
                ]
                score_points = [
                    float(row["xdp_total_budget_xi_point"])
                    for row in score_rows
                    if row["xdp_total_budget_xi_point"] is not None
                ]
                score_confidence = [
                    row["xdp_total_budget_xi_confidence"]
                    for row in score_rows
                ]
                confidence_resolved = all(
                    value is not None for value in score_confidence
                )
                output.append(
                    {
                        "angular_radius": radius,
                        "method": method,
                        "delta": float(delta),
                        "status": (
                            "resolved"
                            if confidence_resolved
                            else "point_only_insufficient_confidence_samples"
                        ),
                        "evaluated_pairs": len(eligible),
                        "linkage_xdp_xi_point_envelope": max(linkage_values),
                        "score_xdp_xi_point_envelope": (
                            None if not score_points else max(score_points)
                        ),
                        "score_xdp_xi_confidence_envelope": (
                            max(float(value) for value in score_confidence)
                            if confidence_resolved
                            else None
                        ),
                    }
                )
    return {
        "definition": (
            "maximum pair-specific empirical XDP budget over evaluated real "
            "pairs whose normalized angular distance does not exceed the "
            "declared radius"
        ),
        "rows": output,
    }
