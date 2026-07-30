"""Empirical angular-distance envelope for a fixed pretrained IsoHash.

For each declared radius ``r``, this module defines an explicit evaluation
distribution:

1. choose an anchor uniformly from anchors having at least one pool neighbor
   with normalized angular distance at most ``r``;
2. choose one such neighbor uniformly;
3. choose a uniform ordered route without replacement;
4. calculate the aggregate fixed-IsoHash Hamming distance ``U`` in either one
   selected replica or the complete t-replica server view;
5. optionally sample the corresponding randomized-response privacy loss
   ``L=(2Z-U) eta``, where ``Z ~ Binomial(U,1-p)``.

The samples are converted into a distribution-free one-sided tolerance limit.
With confidence at least ``1-beta``, the resulting threshold ``u`` satisfies

    Pr[U > u | d_theta(x,x') <= r] <= delta

under the declared finite-pool sampling distribution.  Multiplying ``u`` by
the randomized-response per-bit cost gives an empirical pure-loss envelope.
The same order-statistic construction applied directly to sampled privacy
losses gives ``Pr[L > epsilon] <= delta``.  Both are intentionally labelled
empirical: neither is a substitute for a mechanism-level angular-XDP theorem
over an unrestricted input domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import binom

from .data import IsoHashModel, hash_batch, normalized_rows
from .xdp import rr_epsilon_bit


@dataclass(frozen=True)
class ToleranceLimit:
    sample_count: int
    tail_probability: float
    confidence_beta: float
    requested_confidence: float
    required_samples_for_maximum: int
    sufficient_samples: bool
    order_statistic_rank: Optional[int]
    upper_limit: Optional[float]
    achieved_confidence: float
    diagnostic_observed_maximum: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def required_samples_for_maximum(
    tail_probability: float, confidence_beta: float
) -> int:
    """Samples needed for the observed maximum to be a tolerance bound."""

    delta = float(tail_probability)
    beta = float(confidence_beta)
    if not 0.0 < delta < 1.0:
        raise ValueError("tail_probability must lie in (0,1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("confidence_beta must lie in (0,1)")
    return int(math.ceil(math.log(beta) / math.log(1.0 - delta)))


def one_sided_tolerance_limit(
    samples: Sequence[Union[int, float]],
    tail_probability: float,
    confidence_beta: float,
) -> ToleranceLimit:
    """Distribution-free upper tolerance limit from order statistics.

    If ``U_(k)`` is the k-th ascending order statistic, confidence that it
    covers at least ``1-delta`` probability mass is

        Pr[Binomial(n, 1-delta) <= k-1].

    The smallest feasible rank is used.  If even the sample maximum is
    insufficient, no confident numerical limit is returned.
    """

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("samples cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    delta = float(tail_probability)
    beta = float(confidence_beta)
    if not 0.0 < delta < 1.0:
        raise ValueError("tail_probability must lie in (0,1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("confidence_beta must lie in (0,1)")

    n = int(values.size)
    coverage = 1.0 - delta
    requested = 1.0 - beta
    maximum_confidence = float(binom.cdf(n - 1, n, coverage))
    required = required_samples_for_maximum(delta, beta)
    observed_maximum = float(np.max(values))
    if maximum_confidence + 1e-14 < requested:
        return ToleranceLimit(
            sample_count=n,
            tail_probability=delta,
            confidence_beta=beta,
            requested_confidence=requested,
            required_samples_for_maximum=required,
            sufficient_samples=False,
            order_statistic_rank=None,
            upper_limit=None,
            achieved_confidence=maximum_confidence,
            diagnostic_observed_maximum=observed_maximum,
        )

    rank = None
    achieved = 0.0
    for candidate in range(1, n + 1):
        confidence = float(binom.cdf(candidate - 1, n, coverage))
        if confidence + 1e-14 >= requested:
            rank = candidate
            achieved = confidence
            break
    if rank is None:
        raise RuntimeError("failed to find a feasible tolerance rank")
    ordered = np.sort(values)
    return ToleranceLimit(
        sample_count=n,
        tail_probability=delta,
        confidence_beta=beta,
        requested_confidence=requested,
        required_samples_for_maximum=required,
        sufficient_samples=True,
        order_statistic_rank=rank,
        upper_limit=float(ordered[rank - 1]),
        achieved_confidence=achieved,
        diagnostic_observed_maximum=observed_maximum,
    )


@dataclass
class AngularPool:
    pool_ids: np.ndarray
    normalized_pool: np.ndarray
    anchor_positions: np.ndarray
    neighbor_positions: List[np.ndarray]
    neighbor_distances: List[np.ndarray]


def prepare_angular_pool(
    vectors: np.ndarray,
    maximum_radius: float,
    pool_size: int,
    anchor_count: int,
    seed: int,
) -> AngularPool:
    """Precompute sorted neighbor lists up to the largest requested radius."""

    radius = float(maximum_radius)
    if not 0.0 < radius <= 1.0:
        raise ValueError("maximum_radius must lie in (0,1]")
    rng = np.random.default_rng(int(seed))
    size = min(int(pool_size), int(vectors.shape[0]))
    selected_ids = rng.choice(vectors.shape[0], size=size, replace=False)
    normalized, valid = normalized_rows(vectors[selected_ids])
    selected_ids = selected_ids[valid]
    normalized = normalized[valid]
    if normalized.shape[0] < 2:
        raise ValueError("not enough nonzero vectors for angular evaluation")
    anchors = min(int(anchor_count), int(normalized.shape[0]))
    anchor_positions = rng.choice(
        normalized.shape[0], size=anchors, replace=False
    ).astype(np.int64)

    neighbor_positions = []
    neighbor_distances = []
    block_size = 32
    for start in range(0, anchors, block_size):
        positions = anchor_positions[start : start + block_size]
        cosine = np.clip(normalized[positions] @ normalized.T, -1.0, 1.0)
        distances = np.arccos(cosine) / math.pi
        for local_row, anchor_position in enumerate(positions.tolist()):
            row = distances[local_row]
            mask = (row > 1e-12) & (row <= radius)
            mask[int(anchor_position)] = False
            candidates = np.flatnonzero(mask)
            if candidates.size:
                order = np.argsort(row[candidates], kind="mergesort")
                candidates = candidates[order].astype(np.int32)
                candidate_distances = row[candidates].astype(np.float32)
            else:
                candidates = np.asarray([], dtype=np.int32)
                candidate_distances = np.asarray([], dtype=np.float32)
            neighbor_positions.append(candidates)
            neighbor_distances.append(candidate_distances)
    return AngularPool(
        pool_ids=selected_ids.astype(np.int64),
        normalized_pool=normalized,
        anchor_positions=anchor_positions,
        neighbor_positions=neighbor_positions,
        neighbor_distances=neighbor_distances,
    )


def sample_pairs_within_radius(
    pool: AngularPool,
    radius: float,
    trials: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Draw independent anchor-uniform neighbor pairs from one radius."""

    radius = float(radius)
    trials = int(trials)
    if trials < 1:
        raise ValueError("trials must be positive")
    eligible = []
    eligible_counts = []
    for anchor_index, distances in enumerate(pool.neighbor_distances):
        count = int(np.searchsorted(distances, radius, side="right"))
        if count > 0:
            eligible.append(anchor_index)
            eligible_counts.append(count)
    if not eligible:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
            0,
        )

    rng = np.random.default_rng(int(seed))
    selected_eligible = rng.integers(0, len(eligible), size=trials)
    first_ids = np.empty(trials, dtype=np.int64)
    second_ids = np.empty(trials, dtype=np.int64)
    actual_distances = np.empty(trials, dtype=np.float64)
    for trial, eligible_position in enumerate(selected_eligible.tolist()):
        anchor_list_index = eligible[int(eligible_position)]
        count = eligible_counts[int(eligible_position)]
        neighbor_offset = int(rng.integers(0, count))
        anchor_pool_position = int(pool.anchor_positions[anchor_list_index])
        neighbor_pool_position = int(
            pool.neighbor_positions[anchor_list_index][neighbor_offset]
        )
        first_ids[trial] = int(pool.pool_ids[anchor_pool_position])
        second_ids[trial] = int(pool.pool_ids[neighbor_pool_position])
        actual_distances[trial] = float(
            pool.neighbor_distances[anchor_list_index][neighbor_offset]
        )
    return first_ids, second_ids, actual_distances, len(eligible)


def sample_route_mismatch_curves(
    vectors: np.ndarray,
    model: IsoHashModel,
    first_ids: np.ndarray,
    second_ids: np.ndarray,
    observed_shard_counts: Sequence[int],
    seed: int,
    normalize_centered: bool,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Sample routes once and calculate cumulative mismatch curves.

    The first ``c`` entries of each uniformly random shard permutation define
    a uniform c-of-M route.  This lets the one-replica diagnostic and complete
    t-replica view share the same sampled pair without recomputing hashes.
    """

    first_ids = np.asarray(first_ids, dtype=np.int64)
    second_ids = np.asarray(second_ids, dtype=np.int64)
    if first_ids.shape != second_ids.shape:
        raise ValueError("pair endpoint arrays must have equal shapes")
    trials = first_ids.size
    num_shards = len(model.weights)
    counts = sorted(set(int(value) for value in observed_shard_counts))
    if not counts or counts[0] < 1 or counts[-1] > num_shards:
        raise ValueError("observed shard counts must lie in [1,num_shards]")
    maximum_count = counts[-1]
    rng = np.random.default_rng(int(seed))
    routes = np.empty((trials, maximum_count), dtype=np.int32)
    for trial in range(trials):
        routes[trial] = rng.choice(
            num_shards, size=maximum_count, replace=False
        )

    totals = {
        count: np.zeros(trials, dtype=np.int32) for count in counts
    }
    for shard in range(num_shards):
        selected_trials, positions = np.nonzero(routes == shard)
        if selected_trials.size == 0:
            continue
        first_codes = hash_batch(
            vectors[first_ids[selected_trials]],
            model.mean,
            model.weights[shard],
            model.thresholds[shard],
            normalize_centered,
        )
        second_codes = hash_batch(
            vectors[second_ids[selected_trials]],
            model.mean,
            model.weights[shard],
            model.thresholds[shard],
            normalize_centered,
        )
        differences = np.count_nonzero(
            first_codes != second_codes, axis=1
        ).astype(np.int32)
        for count in counts:
            included = positions < count
            if np.any(included):
                totals[count][selected_trials[included]] += differences[included]
    return totals, routes


def _privacy_loss_samples(
    mismatches: np.ndarray,
    flip_probability: float,
    seed: int,
) -> np.ndarray:
    """Draw privacy losses in one likelihood-ratio direction.

    Randomized response is symmetric, so the reverse direction has the same
    privacy-loss distribution.
    """

    u = np.asarray(mismatches, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    agreements = rng.binomial(u, 1.0 - float(flip_probability))
    return (2.0 * agreements.astype(np.float64) - u) * rr_epsilon_bit(
        flip_probability
    )


def empirical_angular_envelope(
    vectors: np.ndarray,
    model: IsoHashModel,
    radii: Sequence[float],
    trials_per_radius: int,
    pool_size: int,
    anchor_count: int,
    selected_shards: int,
    flip_probability: float,
    tail_probabilities: Sequence[float],
    simultaneous_beta: float,
    seed: int,
    normalize_centered: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Calculate high-confidence empirical angular privacy envelopes."""

    radii = sorted(set(float(radius) for radius in radii))
    if not radii or radii[0] <= 0.0 or radii[-1] > 1.0:
        raise ValueError("radii must be nonempty and lie in (0,1]")
    tails = [float(value) for value in tail_probabilities]
    if any(not 0.0 < value < 1.0 for value in tails):
        raise ValueError("tail probabilities must lie in (0,1)")
    beta = float(simultaneous_beta)
    if not 0.0 < beta < 1.0:
        raise ValueError("simultaneous_beta must lie in (0,1)")
    observed_shard_counts = sorted(set([1, int(selected_shards)]))
    # Two simultaneous statements are reported for each configuration:
    # a pure-loss envelope based on U and a direct privacy-loss tail based on L.
    comparisons = len(radii) * len(tails) * len(observed_shard_counts) * 2
    per_comparison_beta = beta / comparisons
    eta = rr_epsilon_bit(flip_probability)

    pool = prepare_angular_pool(
        vectors=vectors,
        maximum_radius=max(radii),
        pool_size=pool_size,
        anchor_count=anchor_count,
        seed=seed,
    )
    rows = []
    for radius_index, radius in enumerate(radii):
        if progress_callback is not None:
            progress_callback(
                "angular radius %g: sampling %d real pair/route trials"
                % (radius, int(trials_per_radius))
            )
        first, second, actual_distances, eligible_anchors = sample_pairs_within_radius(
            pool=pool,
            radius=radius,
            trials=trials_per_radius,
            seed=int(seed) + 100003 * (radius_index + 1),
        )
        if first.size == 0:
            rows.append(
                {
                    "angular_radius": radius,
                    "status": "no_eligible_pairs",
                    "eligible_anchors": 0,
                    "trials": 0,
                    "views": [],
                }
            )
            if progress_callback is not None:
                progress_callback(
                    "angular radius %g: no eligible real pairs in the pool"
                    % radius
                )
            continue
        mismatch_curves, _ = sample_route_mismatch_curves(
            vectors=vectors,
            model=model,
            first_ids=first,
            second_ids=second,
            observed_shard_counts=observed_shard_counts,
            seed=int(seed) + 700001 * (radius_index + 1),
            normalize_centered=normalize_centered,
        )
        view_rows = []
        for observed_shards in observed_shard_counts:
            mismatches = mismatch_curves[observed_shards]
            losses = _privacy_loss_samples(
                mismatches=mismatches,
                flip_probability=flip_probability,
                seed=(
                    int(seed)
                    + 900001 * (radius_index + 1)
                    + 7919 * observed_shards
                ),
            )
            tail_rows = []
            for tail_probability in tails:
                mismatch_limit = one_sided_tolerance_limit(
                    samples=mismatches,
                    tail_probability=tail_probability,
                    confidence_beta=per_comparison_beta,
                )
                loss_limit = one_sided_tolerance_limit(
                    samples=losses,
                    tail_probability=tail_probability,
                    confidence_beta=per_comparison_beta,
                )
                mismatch_row = mismatch_limit.to_dict()
                mismatch_row["mismatch_upper_limit"] = (
                    None
                    if mismatch_limit.upper_limit is None
                    else int(round(mismatch_limit.upper_limit))
                )
                mismatch_row["pure_xdp_xi_upper"] = (
                    None
                    if mismatch_limit.upper_limit is None
                    else eta * mismatch_limit.upper_limit
                )
                mismatch_row["status"] = (
                    "certified_empirical_envelope"
                    if mismatch_limit.sufficient_samples
                    else "insufficient_samples"
                )
                loss_row = loss_limit.to_dict()
                loss_row["privacy_loss_epsilon_upper"] = (
                    None
                    if loss_limit.upper_limit is None
                    else max(0.0, loss_limit.upper_limit)
                )
                loss_row["status"] = (
                    "certified_empirical_tail"
                    if loss_limit.sufficient_samples
                    else "insufficient_samples"
                )
                tail_rows.append(
                    {
                        "tail_probability": tail_probability,
                        "pure_loss_from_mismatch": mismatch_row,
                        "direct_privacy_loss_tail": loss_row,
                    }
                )
            view_rows.append(
                {
                    "observed_shards": observed_shards,
                    "view": (
                        "complete_t_replica_server_view"
                        if observed_shards == int(selected_shards)
                        else "single_selected_replica_diagnostic"
                    ),
                    "aggregate_route_mismatches": {
                        "minimum": int(np.min(mismatches)),
                        "median": float(np.median(mismatches)),
                        "mean": float(np.mean(mismatches)),
                        "maximum": int(np.max(mismatches)),
                    },
                    "sampled_privacy_loss": {
                        "minimum": float(np.min(losses)),
                        "median": float(np.median(losses)),
                        "mean": float(np.mean(losses)),
                        "maximum": float(np.max(losses)),
                    },
                    "tail_bounds": tail_rows,
                }
            )
        rows.append(
            {
                "angular_radius": radius,
                "status": "ok",
                "eligible_anchors": eligible_anchors,
                "trials": int(mismatches.size),
                "actual_angular_distance": {
                    "minimum": float(np.min(actual_distances)),
                    "median": float(np.median(actual_distances)),
                    "mean": float(np.mean(actual_distances)),
                    "maximum": float(np.max(actual_distances)),
                },
                "views": view_rows,
            }
        )
        if progress_callback is not None:
            progress_callback(
                "angular radius %g: completed with %d eligible anchors"
                % (radius, eligible_anchors)
            )

    # XDP budget functions are conventionally presented as nondecreasing in
    # distance.  Finite samples at different radii can cross.  Replacing each
    # bound by the running maximum preserves every upper-tail statement and
    # yields a monotone empirical envelope.
    for observed_shards in observed_shard_counts:
        for tail_probability in tails:
            running_pure = None
            running_loss = None
            for radius_row in rows:
                matching_view = next(
                    (
                        view
                        for view in radius_row.get("views", [])
                        if view["observed_shards"] == observed_shards
                    ),
                    None,
                )
                if matching_view is None:
                    continue
                matching_tail = next(
                    tail
                    for tail in matching_view["tail_bounds"]
                    if tail["tail_probability"] == tail_probability
                )
                pure = matching_tail["pure_loss_from_mismatch"]
                loss = matching_tail["direct_privacy_loss_tail"]

                raw_pure = pure["pure_xdp_xi_upper"]
                pure["raw_pure_xdp_xi_upper"] = raw_pure
                if raw_pure is not None:
                    running_pure = (
                        raw_pure
                        if running_pure is None
                        else max(running_pure, raw_pure)
                    )
                    pure["pure_xdp_xi_upper"] = running_pure
                    pure["monotone_adjusted"] = running_pure > raw_pure

                raw_loss = loss["privacy_loss_epsilon_upper"]
                loss["raw_privacy_loss_epsilon_upper"] = raw_loss
                if raw_loss is not None:
                    running_loss = (
                        raw_loss
                        if running_loss is None
                        else max(running_loss, raw_loss)
                    )
                    loss["privacy_loss_epsilon_upper"] = running_loss
                    loss["monotone_adjusted"] = running_loss > raw_loss

    return {
        "definition": (
            "empirical angular-distance privacy envelope for fixed IsoHash: "
            "with simultaneous confidence at least 1-beta, the reported "
            "bounds satisfy Pr[U>u]<=delta and Pr[L>epsilon]<=delta under "
            "the declared anchor-uniform finite-pool distribution"
        ),
        "not_a_global_theorem": True,
        "sampling_distribution": (
            "anchor uniform among eligible anchors; neighbor uniform within "
            "d_theta<=r; uniform ordered route without replacement; "
            "fresh randomized-response loss draw; trials sampled with replacement"
        ),
        "pool_vectors": int(pool.pool_ids.size),
        "anchors_examined": int(pool.anchor_positions.size),
        "trials_per_radius": int(trials_per_radius),
        "simultaneous_confidence_beta": beta,
        "simultaneous_confidence": 1.0 - beta,
        "comparisons": comparisons,
        "bonferroni_beta_per_comparison": per_comparison_beta,
        "epsilon_bit": eta,
        "observed_shard_counts": observed_shard_counts,
        "radii": rows,
    }
