"""Attack-channel leakage evaluation for the unlabelled multi-shard view.

This module closes the gap between cross-shard matching and privacy
evaluation.  It never gives a matching method the target's complete route or
the ground-truth identities of its other replicas.  Instead, the evaluator
gives the method a small declared set of anchor occurrences and a fixed set
of auxiliary seed correspondences.  MDS, topology, GSGM, and their fixed
joint score must recover the remaining occurrences from shard-local codes and
graph structure.

For a neighbouring pair (x0, x1), every recovered cluster is converted into
one randomized-response log-likelihood score.  Repeated world-0 and world-1
trials therefore produce attack-output distributions.  These distributions
are summarized using AUC, threshold advantage, and a discretized empirical
hockey-stick profile.

The number of ranked matches added to the anchor is selected from a declared
finite grid on independent calibration trials and is frozen before testing.
This models a stronger rational adversary that can ignore low-confidence
matches instead of being forced to aggregate all t-1 candidates.

The resulting profile applies only to the declared attack channel.  It is not
an unrestricted mechanism-level DP guarantee.  The oracle-linked analytical
accountant remains a separate conservative reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_distribution

from .angular_envelope import one_sided_tolerance_limit
from .data import ChallengePair, IsoHashModel, hash_batch
from .linkage import (
    GraphView,
    _align_with_seeds,
    _classical_mds,
    _graph_signal_features,
    _negative_squared_distance,
    _score_standardize,
    _standardize_pair,
    _topology_features,
    build_graph_views,
)
from .reconstruction_xdp import (
    angular_xdp_envelope,
    linkage_aware_xdp,
    reconstructed_score_as_xdp,
)
from .xdp import rr_epsilon_bit


METHODS = ("anchor", "mds", "topology", "gsgm", "joint")


@dataclass(frozen=True)
class Candidate:
    shard: int
    row: int
    confidence: float


def _count_histogram(values: Sequence[float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(int(round(float(value))))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _empirical_upper_quantile(
    values: Sequence[float],
    tail_probability: float,
) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("quantile values cannot be empty")
    coverage = 1.0 - float(tail_probability)
    index = max(
        0,
        min(
            int(ordered.size) - 1,
            int(math.ceil(coverage * ordered.size)) - 1,
        ),
    )
    return float(ordered[index])


def _linked_replica_xdp_exposure(
    replicas_world0: Sequence[float],
    replicas_world1: Sequence[float],
    single_shard_xdp: float,
    single_shard_delta: float,
    tail_probabilities: Sequence[float],
    confidence_beta_per_limit: float,
) -> Dict[str, object]:
    """Map correctly linked replicas to a single-shard-XDP composition scale.

    The calculation deliberately uses only the declared single-shard
    ``(xi_1, delta_1)`` guarantee. Basic composition maps ``r`` correctly
    associated shard releases to ``(r*xi_1, r*delta_1)``. A full single-shard
    PLD is required for a tighter fixed-total-delta composition.
    """

    by_world = {
        0: np.asarray(replicas_world0, dtype=np.float64),
        1: np.asarray(replicas_world1, dtype=np.float64),
    }
    combined = np.concatenate([by_world[0], by_world[1]])
    xi1 = float(single_shard_xdp)
    delta1 = float(single_shard_delta)
    rows = []
    for tail in tail_probabilities:
        limits = {
            world: one_sided_tolerance_limit(
                values,
                tail_probability=float(tail),
                confidence_beta=float(confidence_beta_per_limit),
            )
            for world, values in by_world.items()
        }
        sufficient = all(value.sufficient_samples for value in limits.values())
        if sufficient:
            replica_limit = int(
                max(
                    round(float(value.upper_limit))
                    for value in limits.values()
                    if value.upper_limit is not None
                )
            )
            xi_limit = xi1 * replica_limit
            composed_delta = min(1.0, delta1 * replica_limit)
        else:
            replica_limit = None
            xi_limit = None
            composed_delta = None
        diagnostic_max = int(
            max(value.diagnostic_observed_maximum for value in limits.values())
        )
        rows.append(
            {
                "tail_probability": float(tail),
                "status": "resolved" if sufficient else "insufficient_samples",
                "replica_count_upper_limit": replica_limit,
                "xdp_upper_limit": xi_limit,
                "basic_composition_delta": composed_delta,
                "diagnostic_observed_max_replicas": diagnostic_max,
                "diagnostic_observed_max_xdp": xi1 * diagnostic_max,
                "empirical_replica_quantile": max(
                    _empirical_upper_quantile(values, float(tail))
                    for values in by_world.values()
                ),
                "world0_tolerance": limits[0].to_dict(),
                "world1_tolerance": limits[1].to_dict(),
            }
        )
    return {
        "definition": (
            "attack-linked multi-shard XDP exposure obtained by basic "
            "composition of the declared single-shard XDP guarantee over "
            "correctly associated target replicas"
        ),
        "single_shard_xdp": xi1,
        "single_shard_delta": delta1,
        "composition_rule": (
            "r correctly linked releases map to "
            "(r*single_shard_xdp, r*single_shard_delta)-XDP"
        ),
        "fixed_total_delta_note": (
            "a tighter composition at one fixed total delta requires the "
            "complete single-shard privacy-loss distribution, not only one "
            "(xi,delta) point"
        ),
        "mean_correctly_linked_replicas": float(combined.mean()),
        "mean_xdp_exposure": float(xi1 * combined.mean()),
        "median_xdp_exposure": float(xi1 * np.median(combined)),
        "replica_count_histogram": _count_histogram(combined),
        "world0_replica_count_histogram": _count_histogram(by_world[0]),
        "world1_replica_count_histogram": _count_histogram(by_world[1]),
        "tail_rows": rows,
    }


def _rr_log_likelihood_ratio(
    report: np.ndarray,
    clean_world0: np.ndarray,
    clean_world1: np.ndarray,
    epsilon_bit: float,
) -> float:
    """Return log Pr[report|x0] / Pr[report|x1] for symmetric RR."""

    report = np.asarray(report, dtype=np.uint8)
    clean0 = np.asarray(clean_world0, dtype=np.uint8)
    clean1 = np.asarray(clean_world1, dtype=np.uint8)
    if report.shape != clean0.shape or report.shape != clean1.shape:
        raise ValueError("report and clean codes must have identical shapes")
    # Coordinates on which the clean codes agree cancel from the ratio.
    distance0 = int(np.count_nonzero(report != clean0))
    distance1 = int(np.count_nonzero(report != clean1))
    return float(epsilon_bit) * float(distance1 - distance0)


def _method_score_matrices(
    first: GraphView,
    second: GraphView,
    seeds: Sequence[int],
    mds_dimensions: int,
    graph_signal_hops: int,
) -> Mapping[str, np.ndarray]:
    first_seed_rows = [first.index[int(value)] for value in seeds]
    second_seed_rows = [second.index[int(value)] for value in seeds]

    mds_first = _classical_mds(first.distances, mds_dimensions)
    mds_second = _classical_mds(second.distances, mds_dimensions)
    mds_first = _align_with_seeds(
        mds_first,
        mds_second,
        first_seed_rows,
        second_seed_rows,
    )
    mds_score = _negative_squared_distance(mds_first, mds_second)

    topology_first, topology_second = _standardize_pair(
        _topology_features(first),
        _topology_features(second),
    )
    topology_score = _negative_squared_distance(
        topology_first,
        topology_second,
    )

    signal_first, signal_second = _standardize_pair(
        _graph_signal_features(first, seeds, graph_signal_hops),
        _graph_signal_features(second, seeds, graph_signal_hops),
    )
    gsgm_score = _negative_squared_distance(signal_first, signal_second)

    joint_score = (
        _score_standardize(mds_score)
        + _score_standardize(topology_score)
        + _score_standardize(gsgm_score)
    ) / 3.0
    return {
        "mds": mds_score,
        "topology": topology_score,
        "gsgm": gsgm_score,
        "joint": joint_score,
    }


def _best_candidate_from_anchor_rows(
    score_rows: Sequence[np.ndarray],
    destination: GraphView,
    excluded_ids: Sequence[int],
) -> Optional[Candidate]:
    """Fuse target-to-destination scores from all declared target anchors."""

    excluded = set(int(value) for value in excluded_ids)
    rows = np.asarray(
        [
            row
            for row, logical_id in enumerate(destination.ids.tolist())
            if int(logical_id) not in excluded
        ],
        dtype=np.int64,
    )
    if rows.size == 0 or not score_rows:
        return None

    normalized = []
    for score_row in score_rows:
        values = np.asarray(score_row, dtype=np.float64)[rows]
        if not np.all(np.isfinite(values)):
            continue
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center))) * 1.4826
        if scale < 1e-12:
            scale = float(np.std(values))
        if scale < 1e-12:
            scale = 1.0
        normalized.append((values - center) / scale)
    if not normalized:
        return None

    fused = np.mean(np.vstack(normalized), axis=0)
    order = np.argsort(fused, kind="mergesort")[::-1]
    best_position = int(order[0])
    best_row = int(rows[best_position])
    best = float(fused[best_position])
    second = float(fused[int(order[1])]) if order.size > 1 else best
    scale = float(np.std(fused))
    if scale < 1e-12:
        scale = 1.0
    return Candidate(
        shard=int(destination.shard),
        row=best_row,
        confidence=float((best - second) / scale),
    )


def _clean_pair_codes(
    x0: np.ndarray,
    x1: np.ndarray,
    model: IsoHashModel,
    normalize_centered: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    pair = np.vstack([x0, x1]).astype(np.float32)
    world0 = []
    world1 = []
    for weight, threshold in zip(model.weights, model.thresholds):
        codes = hash_batch(
            pair,
            model.mean,
            weight,
            threshold,
            normalize_centered,
        )
        world0.append(codes[0])
        world1.append(codes[1])
    return world0, world1


def _one_world_trial(
    world_vectors: np.ndarray,
    clean_world0: Sequence[np.ndarray],
    clean_world1: Sequence[np.ndarray],
    model: IsoHashModel,
    selected_shards: int,
    flip_probability: float,
    neighbors: int,
    seed_fraction: float,
    mds_dimensions: int,
    graph_signal_hops: int,
    aggregation_sizes: Sequence[int],
    known_target_anchors: int,
    seed: int,
    normalize_centered: bool,
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, Dict[str, float]]],
]:
    """Run one unlabelled-view trial.

    Logical id zero is the challenge record.  Its identity is revealed in the
    declared number of selected shards; all other challenge occurrences must
    be recovered. Background identities are used only to instantiate the
    declared auxiliary seed correspondences.
    """

    views = build_graph_views(
        vectors=world_vectors,
        model=model,
        selected_shards=selected_shards,
        flip_probability=flip_probability,
        neighbors=neighbors,
        seed=seed,
        normalize_centered=normalize_centered,
    )
    target_views = [view for view in views if 0 in view.index]
    if len(target_views) != int(selected_shards):
        raise RuntimeError("challenge route does not contain selected_shards views")
    anchor_count = int(known_target_anchors)
    if not 1 <= anchor_count <= int(selected_shards):
        raise ValueError(
            "known_target_anchors must lie in [1, selected_shards]"
        )
    # The declared anchor occurrences are explicit auxiliary information.
    # All remaining occurrences and the rest of the route remain hidden.
    ordered_target_views = sorted(target_views, key=lambda view: view.shard)
    known_anchors = ordered_target_views[:anchor_count]
    known_anchor_shards = set(view.shard for view in known_anchors)
    epsilon_bit = rr_epsilon_bit(flip_probability)
    anchor_llr = sum(
        _rr_log_likelihood_ratio(
            view.codes[int(view.index[0])],
            clean_world0[view.shard],
            clean_world1[view.shard],
            epsilon_bit,
        )
        for view in known_anchors
    )
    clean_distances = [
        int(np.count_nonzero(first != second))
        for first, second in zip(clean_world0, clean_world1)
    ]
    anchor_effective_distance = sum(
        clean_distances[view.shard] for view in known_anchors
    )

    candidates: Dict[str, List[Candidate]] = {
        method: [] for method in METHODS if method != "anchor"
    }
    rng = np.random.default_rng(int(seed) + 104729)
    usable_pairs = 0
    for destination in views:
        if (
            destination.shard in known_anchor_shards
            or destination.ids.size == 0
        ):
            continue
        rows_by_method: Dict[str, List[np.ndarray]] = {
            method: [] for method in candidates
        }
        excluded_seed_ids = set()
        for source in known_anchors:
            common = sorted(
                value
                for value in set(source.index).intersection(destination.index)
                if int(value) != 0
            )
            if len(common) < 2:
                continue
            seed_count = max(
                2,
                int(round(len(common) * float(seed_fraction))),
            )
            seed_count = min(seed_count, len(common))
            seeds = sorted(
                int(value)
                for value in rng.choice(
                    np.asarray(common, dtype=np.int64),
                    size=seed_count,
                    replace=False,
                ).tolist()
            )
            excluded_seed_ids.update(seeds)
            matrices = _method_score_matrices(
                source,
                destination,
                seeds,
                mds_dimensions,
                graph_signal_hops,
            )
            source_row = int(source.index[0])
            for method, matrix in matrices.items():
                rows_by_method[method].append(matrix[source_row])
            usable_pairs += 1

        for method, score_rows in rows_by_method.items():
            candidate = _best_candidate_from_anchor_rows(
                score_rows,
                destination,
                sorted(excluded_seed_ids),
            )
            if candidate is not None:
                candidates[method].append(candidate)

    scores = {"anchor": {0: float(anchor_llr)}}
    diagnostics = {
        "anchor": {
            0: {
                "selected_nodes": float(anchor_count),
                "true_replicas": float(anchor_count),
                "effective_hash_distance": float(anchor_effective_distance),
                "false_blocks": 0.0,
                "false_block_normalized_distance_mean": 0.0,
                "usable_graph_pairs": float(usable_pairs),
            }
        }
    }
    maximum_additional = max(0, int(selected_shards) - anchor_count)
    for method in ("mds", "topology", "gsgm", "joint"):
        ordered = sorted(
            candidates[method],
            key=lambda item: (-item.confidence, item.shard, item.row),
        )
        selected = ordered[:maximum_additional]
        prefix_scores = [float(anchor_llr)]
        prefix_true = [anchor_count]
        prefix_effective_distance = [int(anchor_effective_distance)]
        prefix_false_blocks = [0]
        prefix_false_distance_sum = [0.0]
        for candidate in selected:
            view = views[candidate.shard]
            report = view.codes[candidate.row]
            distance0 = int(
                np.count_nonzero(report != clean_world0[candidate.shard])
            )
            distance1 = int(
                np.count_nonzero(report != clean_world1[candidate.shard])
            )
            is_true = int(int(view.ids[candidate.row]) == 0)
            normalized_null_distance = (
                float(distance0 + distance1)
                / float(2 * max(1, report.size))
            )
            prefix_scores.append(
                prefix_scores[-1]
                + _rr_log_likelihood_ratio(
                    report,
                    clean_world0[candidate.shard],
                    clean_world1[candidate.shard],
                    epsilon_bit,
                )
            )
            prefix_true.append(prefix_true[-1] + is_true)
            prefix_effective_distance.append(
                prefix_effective_distance[-1]
                + is_true * clean_distances[candidate.shard]
            )
            prefix_false_blocks.append(
                prefix_false_blocks[-1] + (1 - is_true)
            )
            prefix_false_distance_sum.append(
                prefix_false_distance_sum[-1]
                + (1 - is_true) * normalized_null_distance
            )
        scores[method] = {}
        diagnostics[method] = {}
        for requested in aggregation_sizes:
            take = min(int(requested), len(selected))
            scores[method][int(requested)] = float(prefix_scores[take])
            diagnostics[method][int(requested)] = {
                "selected_nodes": float(anchor_count + take),
                "true_replicas": float(prefix_true[take]),
                "effective_hash_distance": float(
                    prefix_effective_distance[take]
                ),
                "false_blocks": float(prefix_false_blocks[take]),
                "false_block_normalized_distance_mean": (
                    0.0
                    if prefix_false_blocks[take] == 0
                    else float(prefix_false_distance_sum[take])
                    / float(prefix_false_blocks[take])
                ),
                "usable_graph_pairs": float(usable_pairs),
            }
    return scores, diagnostics


def _auc(world0: np.ndarray, world1: np.ndarray) -> float:
    """Return P[T0>T1] + 0.5 P[T0=T1] without external ML packages."""

    first = np.asarray(world0, dtype=np.float64)
    second = np.asarray(world1, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return float("nan")
    comparisons = first[:, None] - second[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / float(comparisons.size)
    )


def _threshold_advantage(world0: np.ndarray, world1: np.ndarray) -> float:
    first = np.asarray(world0, dtype=np.float64)
    second = np.asarray(world1, dtype=np.float64)
    thresholds = np.unique(np.concatenate([first, second]))
    if thresholds.size == 0:
        return float("nan")
    values = [
        abs(float(np.mean(first >= threshold)) - float(np.mean(second >= threshold)))
        for threshold in thresholds
    ]
    return float(max(values))


def _calibration_edges(
    world0: Sequence[float],
    world1: Sequence[float],
    requested_bins: int,
) -> np.ndarray:
    pooled = np.asarray(list(world0) + list(world1), dtype=np.float64)
    if pooled.size == 0:
        raise ValueError("calibration scores are empty")
    bin_count = max(2, int(requested_bins))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    internal = np.unique(np.quantile(pooled, quantiles))
    return np.concatenate(([-np.inf], internal, [np.inf]))


def _histogram(scores: Sequence[float], edges: np.ndarray) -> np.ndarray:
    # np.histogram accepts infinite outer edges and returns len(edges)-1 bins.
    counts, _ = np.histogram(np.asarray(scores, dtype=np.float64), bins=edges)
    return counts.astype(np.int64)


def _clopper_pearson(
    count: int,
    total: int,
    alpha: float,
) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    lower = (
        0.0
        if count == 0
        else float(beta_distribution.ppf(alpha / 2.0, count, total - count + 1))
    )
    upper = (
        1.0
        if count == total
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0,
                count + 1,
                total - count,
            )
        )
    )
    return lower, upper


def _hockey_stick(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float,
) -> float:
    scale = math.exp(min(float(epsilon), 700.0))
    forward = float(np.maximum(first - scale * second, 0.0).sum())
    reverse = float(np.maximum(second - scale * first, 0.0).sum())
    return min(1.0, max(forward, reverse))


def _epsilon_for_delta(
    profile: Callable[[float], float],
    target_delta: float,
    maximum_epsilon: float = 5000.0,
) -> Optional[float]:
    delta = float(target_delta)
    if profile(0.0) <= delta:
        return 0.0
    high = 1.0
    while high < maximum_epsilon and profile(high) > delta:
        high *= 2.0
    high = min(high, maximum_epsilon)
    if profile(high) > delta:
        return None
    low = 0.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if profile(middle) <= delta:
            high = middle
        else:
            low = middle
    return float(high)


def _attack_profile(
    calibration_world0: Sequence[float],
    calibration_world1: Sequence[float],
    test_world0: Sequence[float],
    test_world1: Sequence[float],
    requested_bins: int,
    deltas: Sequence[float],
    profile_beta: float,
) -> Dict[str, object]:
    edges = _calibration_edges(
        calibration_world0,
        calibration_world1,
        requested_bins,
    )
    count0 = _histogram(test_world0, edges)
    count1 = _histogram(test_world1, edges)
    n0 = int(count0.sum())
    n1 = int(count1.sum())
    bins = int(count0.size)

    # Jeffreys smoothing is used only for a finite descriptive point estimate.
    # The confidence envelope below uses unsmoothed binomial intervals.
    point0 = (count0.astype(np.float64) + 0.5) / (n0 + 0.5 * bins)
    point1 = (count1.astype(np.float64) + 0.5) / (n1 + 0.5 * bins)

    cell_alpha = float(profile_beta) / float(max(1, 2 * bins))
    intervals0 = [_clopper_pearson(int(value), n0, cell_alpha) for value in count0]
    intervals1 = [_clopper_pearson(int(value), n1, cell_alpha) for value in count1]
    lower0 = np.asarray([value[0] for value in intervals0])
    upper0 = np.asarray([value[1] for value in intervals0])
    lower1 = np.asarray([value[0] for value in intervals1])
    upper1 = np.asarray([value[1] for value in intervals1])

    def point_profile(epsilon: float) -> float:
        return _hockey_stick(point0, point1, epsilon)

    def upper_profile(epsilon: float) -> float:
        scale = math.exp(min(float(epsilon), 700.0))
        forward = float(np.maximum(upper0 - scale * lower1, 0.0).sum())
        reverse = float(np.maximum(upper1 - scale * lower0, 0.0).sum())
        return min(1.0, max(forward, reverse))

    delta_rows = []
    for delta in deltas:
        point_epsilon = _epsilon_for_delta(point_profile, float(delta))
        upper_epsilon = _epsilon_for_delta(upper_profile, float(delta))
        delta_rows.append(
            {
                "delta": float(delta),
                "epsilon_point": point_epsilon,
                "epsilon_confidence_upper": upper_epsilon,
                "confidence_status": (
                    "resolved" if upper_epsilon is not None else "insufficient_trials"
                ),
            }
        )

    profile_epsilons = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    return {
        "definition": (
            "discretized attack-output channel; bin edges fixed on independent "
            "calibration trials"
        ),
        "bins": bins,
        "bin_edges": [
            None if not math.isfinite(float(value)) else float(value)
            for value in edges
        ],
        "test_counts_world0": count0.tolist(),
        "test_counts_world1": count1.tolist(),
        "profile_confidence": 1.0 - float(profile_beta),
        "delta_rows": delta_rows,
        "profile_rows": [
            {
                "epsilon": epsilon,
                "delta_point": point_profile(epsilon),
                "delta_confidence_upper": upper_profile(epsilon),
            }
            for epsilon in profile_epsilons
        ],
    }


def evaluate_attack_channels(
    vectors: np.ndarray,
    model: IsoHashModel,
    challenge_pairs: Sequence[ChallengePair],
    background_records: int,
    calibration_runs_per_world: int,
    test_runs_per_world: int,
    selected_shards: int,
    flip_probability: float,
    neighbors: int,
    seed_fraction: float,
    mds_dimensions: int,
    graph_signal_hops: int,
    aggregation_sizes: Sequence[int],
    known_target_anchors: int,
    single_shard_xdp: float,
    single_shard_delta: float,
    exposure_tail_probabilities: Sequence[float],
    score_bins: int,
    deltas: Sequence[float],
    angular_radii: Sequence[float],
    confidence_beta: float,
    seed: int,
    normalize_centered: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Evaluate attack-output leakage on fixed real challenge pairs."""

    if not challenge_pairs:
        raise ValueError("at least one attack-audit challenge pair is required")
    if calibration_runs_per_world < 2 or test_runs_per_world < 2:
        raise ValueError("at least two calibration and test runs per world are required")
    if not 0.0 < confidence_beta < 1.0:
        raise ValueError("confidence_beta must lie in (0,1)")
    if not angular_radii or any(
        not 0.0 < float(value) <= 1.0 for value in angular_radii
    ):
        raise ValueError("angular_radii must contain values in (0,1]")
    if float(single_shard_xdp) < 0.0:
        raise ValueError("single_shard_xdp cannot be negative")
    if not 0.0 <= float(single_shard_delta) < 1.0:
        raise ValueError("single_shard_delta must lie in [0,1)")
    if not exposure_tail_probabilities or any(
        not 0.0 < float(value) < 1.0
        for value in exposure_tail_probabilities
    ):
        raise ValueError(
            "exposure_tail_probabilities must lie in (0,1)"
        )
    anchor_count = int(known_target_anchors)
    if not 1 <= anchor_count <= int(selected_shards):
        raise ValueError(
            "known_target_anchors must lie in [1, selected_shards]"
        )
    maximum_additional = max(0, int(selected_shards) - anchor_count)
    normalized_sizes = sorted(
        set(
            [0, maximum_additional]
            + [
                min(maximum_additional, max(0, int(value)))
                for value in aggregation_sizes
            ]
        )
    )

    rng = np.random.default_rng(int(seed))
    pair_rows = []
    comparisons = len(challenge_pairs) * len(METHODS)
    # Each method/pair produces both a calibration-selected diagnostic profile
    # and a fixed-length reconstructed-vector profile.
    per_profile_beta = float(confidence_beta) / float(max(1, 2 * comparisons))
    per_exposure_beta = float(confidence_beta) / float(
        max(
            1,
            comparisons * 2 * len(exposure_tail_probabilities),
        )
    )

    for pair_index, pair in enumerate(challenge_pairs):
        excluded = {int(pair.first_id), int(pair.second_id)}
        available = np.asarray(
            [index for index in range(vectors.shape[0]) if index not in excluded],
            dtype=np.int64,
        )
        take = min(int(background_records), int(available.size))
        if take < 8:
            raise ValueError("at least eight background records are required")
        background_ids = rng.choice(available, size=take, replace=False)
        background = np.asarray(vectors[background_ids], dtype=np.float32)
        x0 = np.asarray(vectors[pair.first_id], dtype=np.float32)
        x1 = np.asarray(vectors[pair.second_id], dtype=np.float32)
        clean0, clean1 = _clean_pair_codes(
            x0,
            x1,
            model,
            normalize_centered,
        )

        method_sizes = {
            method: ([0] if method == "anchor" else normalized_sizes)
            for method in METHODS
        }
        calibration_scores = {
            method: {
                size: {0: [], 1: []}
                for size in method_sizes[method]
            }
            for method in METHODS
        }
        test_scores = {
            method: {
                size: {0: [], 1: []}
                for size in method_sizes[method]
            }
            for method in METHODS
        }
        test_diagnostics = {
            method: {
                size: {0: [], 1: []}
                for size in method_sizes[method]
            }
            for method in METHODS
        }

        total_runs = 2 * (calibration_runs_per_world + test_runs_per_world)
        completed = 0
        for phase, runs, destination_scores in (
            ("calibration", calibration_runs_per_world, calibration_scores),
            ("test", test_runs_per_world, test_scores),
        ):
            for world, challenge in ((0, x0), (1, x1)):
                world_vectors = np.vstack([challenge[None, :], background])
                for run_index in range(int(runs)):
                    trial_seed = (
                        int(seed)
                        + pair_index * 1000003
                        + (0 if phase == "calibration" else 400009)
                        + world * 200003
                        + run_index * 1009
                    )
                    scores, diagnostics = _one_world_trial(
                        world_vectors=world_vectors,
                        clean_world0=clean0,
                        clean_world1=clean1,
                        model=model,
                        selected_shards=selected_shards,
                        flip_probability=flip_probability,
                        neighbors=neighbors,
                        seed_fraction=seed_fraction,
                        mds_dimensions=mds_dimensions,
                        graph_signal_hops=graph_signal_hops,
                        aggregation_sizes=normalized_sizes,
                        known_target_anchors=anchor_count,
                        seed=trial_seed,
                        normalize_centered=normalize_centered,
                    )
                    for method in METHODS:
                        for size in method_sizes[method]:
                            destination_scores[method][size][world].append(
                                scores[method][size]
                            )
                            if phase == "test":
                                test_diagnostics[method][size][world].append(
                                    diagnostics[method][size]
                                )
                    completed += 1
                    if (
                        progress_callback is not None
                        and (completed == total_runs or completed % max(1, total_runs // 5) == 0)
                    ):
                        progress_callback(
                            "attack-channel pair %d/%d: %d/%d trials"
                            % (
                                pair_index + 1,
                                len(challenge_pairs),
                                completed,
                                total_runs,
                            )
                        )

        method_rows = {}
        for method in METHODS:
            calibration_rows = []
            for size in method_sizes[method]:
                first = np.asarray(
                    calibration_scores[method][size][0],
                    dtype=np.float64,
                )
                second = np.asarray(
                    calibration_scores[method][size][1],
                    dtype=np.float64,
                )
                calibration_rows.append(
                    {
                        "additional_candidates": int(size),
                        "auc_world0_over_world1": _auc(first, second),
                        "threshold_advantage": _threshold_advantage(
                            first,
                            second,
                        ),
                    }
                )
            # The candidate-count grid is declared before the experiment.
            # Selection uses calibration trials only; the chosen size is then
            # frozen for all held-out test calculations.
            chosen = max(
                calibration_rows,
                key=lambda row: (
                    row["auc_world0_over_world1"],
                    row["threshold_advantage"],
                    -row["additional_candidates"],
                ),
            )
            chosen_size = int(chosen["additional_candidates"])
            world0 = np.asarray(
                test_scores[method][chosen_size][0],
                dtype=np.float64,
            )
            world1 = np.asarray(
                test_scores[method][chosen_size][1],
                dtype=np.float64,
            )
            diagnostics = (
                test_diagnostics[method][chosen_size][0]
                + test_diagnostics[method][chosen_size][1]
            )
            replicas_world0 = np.asarray(
                [
                    row["true_replicas"]
                    for row in test_diagnostics[method][chosen_size][0]
                ],
                dtype=np.float64,
            )
            replicas_world1 = np.asarray(
                [
                    row["true_replicas"]
                    for row in test_diagnostics[method][chosen_size][1]
                ],
                dtype=np.float64,
            )
            true_replicas = np.asarray(
                [row["true_replicas"] for row in diagnostics],
                dtype=np.float64,
            )
            selected_nodes = np.asarray(
                [row["selected_nodes"] for row in diagnostics],
                dtype=np.float64,
            )
            profile = _attack_profile(
                calibration_scores[method][chosen_size][0],
                calibration_scores[method][chosen_size][1],
                world0,
                world1,
                requested_bins=score_bins,
                deltas=deltas,
                profile_beta=per_profile_beta,
            )
            xdp_exposure = _linked_replica_xdp_exposure(
                replicas_world0=replicas_world0,
                replicas_world1=replicas_world1,
                single_shard_xdp=single_shard_xdp,
                single_shard_delta=single_shard_delta,
                tail_probabilities=exposure_tail_probabilities,
                confidence_beta_per_limit=per_exposure_beta,
            )
            reconstruction_size = (
                0 if method == "anchor" else maximum_additional
            )
            reconstruction_world0 = np.asarray(
                test_scores[method][reconstruction_size][0],
                dtype=np.float64,
            )
            reconstruction_world1 = np.asarray(
                test_scores[method][reconstruction_size][1],
                dtype=np.float64,
            )
            reconstruction_diagnostics0 = test_diagnostics[method][
                reconstruction_size
            ][0]
            reconstruction_diagnostics1 = test_diagnostics[method][
                reconstruction_size
            ][1]
            effective_world0 = np.asarray(
                [
                    int(round(row["effective_hash_distance"]))
                    for row in reconstruction_diagnostics0
                ],
                dtype=np.int64,
            )
            effective_world1 = np.asarray(
                [
                    int(round(row["effective_hash_distance"]))
                    for row in reconstruction_diagnostics1
                ],
                dtype=np.int64,
            )
            linkage_xdp = linkage_aware_xdp(
                effective_distances_world0=effective_world0,
                effective_distances_world1=effective_world1,
                flip_probability=flip_probability,
                angular_distance=float(pair.angular_distance),
                deltas=deltas,
                tail_probabilities=exposure_tail_probabilities,
                confidence_beta_per_limit=per_exposure_beta,
            )
            reconstruction_profile = _attack_profile(
                calibration_scores[method][reconstruction_size][0],
                calibration_scores[method][reconstruction_size][1],
                reconstruction_world0,
                reconstruction_world1,
                requested_bins=score_bins,
                deltas=deltas,
                profile_beta=per_profile_beta,
            )
            reconstructed_xdp = reconstructed_score_as_xdp(
                reconstruction_profile,
                angular_distance=float(pair.angular_distance),
            )
            reconstruction_diagnostics = (
                reconstruction_diagnostics0 + reconstruction_diagnostics1
            )
            false_block_count = int(
                round(
                    sum(
                        float(row["false_blocks"])
                        for row in reconstruction_diagnostics
                    )
                )
            )
            false_distance_total = sum(
                float(row["false_blocks"])
                * float(row["false_block_normalized_distance_mean"])
                for row in reconstruction_diagnostics
            )
            false_distance_mean = (
                None
                if false_block_count == 0
                else float(false_distance_total) / float(false_block_count)
            )
            oriented_auc = _auc(world0, world1)
            method_rows[method] = {
                "score_definition": (
                    "RR log-likelihood evidence from the declared anchor and "
                    "a calibration-selected prefix of candidates ranked by "
                    "this matching method"
                ),
                "aggregation_selection": (
                    "additional-candidate count chosen on independent "
                    "calibration trials and frozen before held-out testing"
                ),
                "candidate_count_grid": list(method_sizes[method]),
                "chosen_additional_candidates": chosen_size,
                "calibration_auc": float(
                    chosen["auc_world0_over_world1"]
                ),
                "calibration_candidates": calibration_rows,
                "auc_world0_over_world1": oriented_auc,
                "auc_best_direction": max(oriented_auc, 1.0 - oriented_auc),
                "threshold_advantage": _threshold_advantage(world0, world1),
                "mean_score_world0": float(world0.mean()),
                "mean_score_world1": float(world1.mean()),
                "mean_selected_nodes": float(selected_nodes.mean()),
                "mean_true_replicas_recovered": float(true_replicas.mean()),
                "median_true_replicas_recovered": float(
                    np.median(true_replicas)
                ),
                "linked_replica_xdp_exposure": xdp_exposure,
                "privacy_profile": profile,
                "reconstruction_additional_candidates": reconstruction_size,
                "reconstruction_total_blocks": int(
                    anchor_count + reconstruction_size
                ),
                "reconstruction_rule": (
                    "one candidate block per confidence-ranked shard; incorrect "
                    "blocks are retained as the empirical null rather than "
                    "replaced using ground truth"
                ),
                "failed_block_null_check": {
                    "false_blocks_observed": false_block_count,
                    "expected_normalized_hamming_distance": 0.5,
                    "observed_mean_normalized_distance_to_two_templates": (
                        false_distance_mean
                    ),
                    "interpretation": (
                        "values near 0.5 support the world-independent random "
                        "block approximation"
                    ),
                },
                "linkage_aware_xdp": linkage_xdp,
                "reconstructed_vector_privacy_profile": (
                    reconstruction_profile
                ),
                "reconstructed_vector_xdp": reconstructed_xdp,
            }

        pair_rows.append(
            {
                "pair_index": pair_index,
                "first_id": int(pair.first_id),
                "second_id": int(pair.second_id),
                "group": pair.group,
                "angular_distance": float(pair.angular_distance),
                "background_records": int(take),
                "methods": method_rows,
            }
        )

    global_tail_rows = []
    for tail_index, tail in enumerate(exposure_tail_probabilities):
        method_tail_rows = [
            pair["methods"][method]["linked_replica_xdp_exposure"][
                "tail_rows"
            ][tail_index]
            for pair in pair_rows
            for method in METHODS
        ]
        resolved = all(row["status"] == "resolved" for row in method_tail_rows)
        if resolved:
            replica_limit = max(
                int(row["replica_count_upper_limit"])
                for row in method_tail_rows
            )
            xdp_limit = float(single_shard_xdp) * replica_limit
            composed_delta = min(
                1.0,
                float(single_shard_delta) * replica_limit,
            )
        else:
            replica_limit = None
            xdp_limit = None
            composed_delta = None
        global_tail_rows.append(
            {
                "tail_probability": float(tail),
                "status": "resolved" if resolved else "insufficient_samples",
                "replica_count_upper_limit": replica_limit,
                "xdp_upper_limit": xdp_limit,
                "basic_composition_delta": composed_delta,
            }
        )

    max_mean_exposure = max(
        float(
            pair["methods"][method]["linked_replica_xdp_exposure"][
                "mean_xdp_exposure"
            ]
        )
        for pair in pair_rows
        for method in METHODS
    )
    reconstructed_angular_curve = angular_xdp_envelope(
        pair_rows=pair_rows,
        methods=METHODS,
        angular_radii=angular_radii,
        deltas=deltas,
    )
    return {
        "scope": (
            "empirical leakage of the declared finite attack channels on the "
            "declared real pairs; not a mechanism-level DP upper bound"
        ),
        "target_side_information": (
            "%d challenge occurrences are supplied as anchors; all remaining "
            "replicas and the rest of the route are hidden" % anchor_count
        ),
        "known_target_anchors": anchor_count,
        "single_shard_xdp_baseline": float(single_shard_xdp),
        "single_shard_delta_baseline": float(single_shard_delta),
        "linked_replica_xdp_global": {
            "scope": (
                "maximum over the declared challenge pairs, worlds, and "
                "inference methods"
            ),
            "simultaneous_confidence": 1.0 - float(confidence_beta),
            "max_mean_xdp_exposure": max_mean_exposure,
            "tail_rows": global_tail_rows,
        },
        "seed_policy": (
            "fixed fraction of background correspondences common to each graph "
            "pair, with a minimum of two when available"
        ),
        "aggregation_policy": (
            "choose a prefix size from the declared grid on independent "
            "calibration trials; freeze it for held-out test trials"
        ),
        "aggregation_sizes": normalized_sizes,
        "methods": list(METHODS),
        "calibration_runs_per_world": int(calibration_runs_per_world),
        "test_runs_per_world": int(test_runs_per_world),
        "simultaneous_confidence": 1.0 - float(confidence_beta),
        "reconstructed_angular_xdp_curve": reconstructed_angular_curve,
        "pairs": pair_rows,
    }
