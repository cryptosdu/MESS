"""Deterministic cross-shard inference evaluation.

The routines here do not train a classifier.  They build sampled shard-local
k-NN graphs from independently perturbed codes, expose a fixed fraction of
cross-shard seed correspondences, and evaluate three established evidence
families:

* MDS geometry aligned by the declared seeds;
* local graph-topology descriptors;
* seed-aligned graph-signal propagation (GSGM-style evidence).

The joint method combines fixed standardized scores.  Ground-truth logical
identities are used only after scoring to calculate Hit@1 and MRR.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import IsoHashModel, hash_batch


@dataclass
class GraphView:
    shard: int
    ids: np.ndarray
    codes: np.ndarray
    distances: np.ndarray
    adjacency: np.ndarray
    index: Dict[int, int]


def _hamming_matrix(codes: np.ndarray) -> np.ndarray:
    binary = np.asarray(codes, dtype=np.int16)
    ones = binary.sum(axis=1, dtype=np.int32)
    product = binary.astype(np.int32) @ binary.astype(np.int32).T
    distances = ones[:, None] + ones[None, :] - 2 * product
    return np.asarray(distances, dtype=np.float64)


def _knn_adjacency(distances: np.ndarray, neighbors: int) -> np.ndarray:
    count = distances.shape[0]
    if count <= 1:
        return np.zeros((count, count), dtype=np.float64)
    k = min(int(neighbors), count - 1)
    working = distances.copy()
    np.fill_diagonal(working, np.inf)
    selected = np.argpartition(working, kth=k - 1, axis=1)[:, :k]
    adjacency = np.zeros((count, count), dtype=np.float64)
    rows = np.repeat(np.arange(count), k)
    adjacency[rows, selected.reshape(-1)] = 1.0
    # HNSW layer-0 edges are not exactly a k-NN graph; symmetrization gives a
    # deterministic proxy for studying graph-structural linkage.
    return np.maximum(adjacency, adjacency.T)


def build_graph_views(
    vectors: np.ndarray,
    model: IsoHashModel,
    selected_shards: int,
    flip_probability: float,
    neighbors: int,
    seed: int,
    normalize_centered: bool,
) -> List[GraphView]:
    count = vectors.shape[0]
    num_shards = len(model.weights)
    t = int(selected_shards)
    if not 1 <= t <= num_shards:
        raise ValueError("selected_shards must lie in [1,num_shards]")
    rng = np.random.default_rng(int(seed))
    routes = [
        rng.choice(num_shards, size=t, replace=False).astype(np.int64)
        for _ in range(count)
    ]
    shard_ids = [[] for _ in range(num_shards)]
    for logical_id, route in enumerate(routes):
        for shard in route.tolist():
            shard_ids[int(shard)].append(logical_id)

    views = []
    for shard in range(num_shards):
        ids = np.asarray(shard_ids[shard], dtype=np.int64)
        if ids.size == 0:
            codes = np.zeros((0, model.thresholds[shard].size), dtype=np.uint8)
            distances = np.zeros((0, 0), dtype=np.float64)
            adjacency = np.zeros((0, 0), dtype=np.float64)
        else:
            clean = hash_batch(
                vectors[ids],
                model.mean,
                model.weights[shard],
                model.thresholds[shard],
                normalize_centered,
            )
            noise = rng.random(clean.shape) < float(flip_probability)
            codes = np.bitwise_xor(clean, noise.astype(np.uint8))
            distances = _hamming_matrix(codes)
            adjacency = _knn_adjacency(distances, neighbors)
        views.append(
            GraphView(
                shard=shard,
                ids=ids,
                codes=codes,
                distances=distances,
                adjacency=adjacency,
                index={int(value): position for position, value in enumerate(ids.tolist())},
            )
        )
    return views


def _classical_mds(distances: np.ndarray, dimensions: int) -> np.ndarray:
    count = distances.shape[0]
    if count == 0:
        return np.zeros((0, int(dimensions)), dtype=np.float64)
    if count == 1:
        return np.zeros((1, int(dimensions)), dtype=np.float64)
    squared = np.square(distances.astype(np.float64))
    centering = np.eye(count) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    positive = [int(index) for index in order if values[index] > 1e-10]
    take = min(int(dimensions), len(positive))
    coordinates = np.zeros((count, int(dimensions)), dtype=np.float64)
    if take:
        chosen = np.asarray(positive[:take], dtype=np.int64)
        coordinates[:, :take] = vectors[:, chosen] * np.sqrt(values[chosen])[None, :]
    return coordinates


def _align_with_seeds(
    source: np.ndarray,
    destination: np.ndarray,
    source_seed_rows: Sequence[int],
    destination_seed_rows: Sequence[int],
) -> np.ndarray:
    if len(source_seed_rows) < 2:
        return source
    source_seed = source[np.asarray(source_seed_rows, dtype=np.int64)]
    destination_seed = destination[np.asarray(destination_seed_rows, dtype=np.int64)]
    source_center = source_seed.mean(axis=0)
    destination_center = destination_seed.mean(axis=0)
    covariance = (source_seed - source_center).T @ (
        destination_seed - destination_center
    )
    left, _, right = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right
    return (source - source_center) @ rotation + destination_center


def _topology_features(view: GraphView) -> np.ndarray:
    adjacency = view.adjacency
    count = adjacency.shape[0]
    if count == 0:
        return np.zeros((0, 5), dtype=np.float64)
    degree = adjacency.sum(axis=1)
    mean_neighbor_degree = np.zeros(count, dtype=np.float64)
    std_neighbor_degree = np.zeros(count, dtype=np.float64)
    two_hop = np.zeros(count, dtype=np.float64)
    clustering = np.zeros(count, dtype=np.float64)
    for node in range(count):
        neighbors = np.flatnonzero(adjacency[node] > 0)
        if neighbors.size:
            neighbor_degrees = degree[neighbors]
            mean_neighbor_degree[node] = float(neighbor_degrees.mean())
            std_neighbor_degree[node] = float(neighbor_degrees.std())
            reached = np.flatnonzero(adjacency[neighbors].sum(axis=0) > 0)
            two_hop[node] = float(max(0, reached.size - 1))
        if neighbors.size >= 2:
            edges = adjacency[np.ix_(neighbors, neighbors)].sum() / 2.0
            clustering[node] = 2.0 * edges / (
                neighbors.size * (neighbors.size - 1)
            )
    denominator = max(1.0, float(count - 1))
    return np.column_stack(
        [
            degree / denominator,
            mean_neighbor_degree / denominator,
            std_neighbor_degree / denominator,
            two_hop / denominator,
            clustering,
        ]
    )


def _graph_signal_features(
    view: GraphView, ordered_seeds: Sequence[int], hops: int
) -> np.ndarray:
    count = view.ids.size
    seed_count = len(ordered_seeds)
    if count == 0:
        return np.zeros((0, max(1, seed_count * int(hops))), dtype=np.float64)
    signal = np.zeros((count, seed_count), dtype=np.float64)
    for column, logical_id in enumerate(ordered_seeds):
        row = view.index.get(int(logical_id))
        if row is not None:
            signal[row, column] = 1.0
    transition = view.adjacency + np.eye(count)
    transition /= np.maximum(transition.sum(axis=1, keepdims=True), 1.0)
    features = []
    current = signal
    for _ in range(max(1, int(hops))):
        current = transition @ current
        features.append(current.copy())
    return np.concatenate(features, axis=1)


def _standardize_pair(
    first: np.ndarray, second: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([first, second])
    mean = combined.mean(axis=0)
    standard_deviation = combined.std(axis=0)
    standard_deviation[standard_deviation < 1e-10] = 1.0
    return (first - mean) / standard_deviation, (
        second - mean
    ) / standard_deviation


def _negative_squared_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.sum(first * first, axis=1, keepdims=True)
    second_norm = np.sum(second * second, axis=1, keepdims=True).T
    return -(first_norm + second_norm - 2.0 * (first @ second.T))


def _score_standardize(matrix: np.ndarray) -> np.ndarray:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.zeros_like(matrix)
    center = float(np.median(finite))
    scale = float(np.median(np.abs(finite - center))) * 1.4826
    if scale < 1e-12:
        scale = float(np.std(finite))
    if scale < 1e-12:
        scale = 1.0
    return (matrix - center) / scale


def _evaluate_score(
    score: np.ndarray,
    source: GraphView,
    destination: GraphView,
    seeds: Sequence[int],
) -> Tuple[Dict[str, float], Dict[int, int]]:
    seed_set = set(int(value) for value in seeds)
    destination_candidates = [
        row
        for row, logical_id in enumerate(destination.ids.tolist())
        if int(logical_id) not in seed_set
    ]
    destination_candidate_array = np.asarray(destination_candidates, dtype=np.int64)
    reciprocal_ranks = []
    hit1 = 0
    predictions = {}
    for source_row, logical_id in enumerate(source.ids.tolist()):
        logical_id = int(logical_id)
        if logical_id in seed_set or logical_id not in destination.index:
            continue
        if destination_candidate_array.size == 0:
            continue
        values = score[source_row, destination_candidate_array]
        order = np.argsort(values, kind="mergesort")[::-1]
        ordered_rows = destination_candidate_array[order]
        predicted_row = int(ordered_rows[0])
        predictions[logical_id] = int(destination.ids[predicted_row])
        true_row = destination.index[logical_id]
        matches = np.flatnonzero(ordered_rows == true_row)
        if matches.size:
            rank = int(matches[0]) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hit1 += 1
    trials = len(reciprocal_ranks)
    metrics = {
        "trials": float(trials),
        "hit_at_1": 0.0 if trials == 0 else hit1 / float(trials),
        "mrr": 0.0 if trials == 0 else float(np.mean(reciprocal_ranks)),
    }
    return metrics, predictions


def _choose_graph_pairs(
    views: Sequence[GraphView], maximum_pairs: int, minimum_overlap: int
) -> List[Tuple[int, int, int]]:
    rows = []
    for first in range(len(views)):
        first_ids = set(int(value) for value in views[first].ids.tolist())
        for second in range(first + 1, len(views)):
            overlap = len(
                first_ids.intersection(int(value) for value in views[second].ids.tolist())
            )
            if overlap >= int(minimum_overlap):
                rows.append((overlap, first, second))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(first, second, overlap) for overlap, first, second in rows[: int(maximum_pairs)]]


def evaluate_linkage(
    vectors: np.ndarray,
    model: IsoHashModel,
    selected_shards: int,
    flip_probability: float,
    neighbors: int,
    seed_fraction: float,
    maximum_graph_pairs: int,
    mds_dimensions: int,
    graph_signal_hops: int,
    seed: int,
    normalize_centered: bool = False,
) -> Dict[str, object]:
    """Run deterministic MDS, topology, GSGM, and joint linkage evaluation."""

    views = build_graph_views(
        vectors=vectors,
        model=model,
        selected_shards=selected_shards,
        flip_probability=flip_probability,
        neighbors=neighbors,
        seed=seed,
        normalize_centered=normalize_centered,
    )
    pairs = _choose_graph_pairs(
        views,
        maximum_pairs=maximum_graph_pairs,
        minimum_overlap=4,
    )
    rng = np.random.default_rng(int(seed) + 7717)
    totals = {
        method: {"trials": 0.0, "hit_sum": 0.0, "mrr_sum": 0.0}
        for method in ("mds", "topology", "gsgm", "joint")
    }
    pair_rows = []
    all_predictions = {}

    for first_index, second_index, overlap_count in pairs:
        first = views[first_index]
        second = views[second_index]
        common = sorted(set(first.index).intersection(second.index))
        seed_count = max(2, int(round(len(common) * float(seed_fraction))))
        seed_count = min(seed_count, max(2, len(common) - 1))
        if len(common) <= seed_count:
            continue
        seeds = sorted(
            int(value)
            for value in rng.choice(
                np.asarray(common, dtype=np.int64), size=seed_count, replace=False
            ).tolist()
        )
        first_seed_rows = [first.index[value] for value in seeds]
        second_seed_rows = [second.index[value] for value in seeds]

        mds_first = _classical_mds(first.distances, mds_dimensions)
        mds_second = _classical_mds(second.distances, mds_dimensions)
        mds_first = _align_with_seeds(
            mds_first, mds_second, first_seed_rows, second_seed_rows
        )
        mds_score = _negative_squared_distance(mds_first, mds_second)

        topology_first, topology_second = _standardize_pair(
            _topology_features(first), _topology_features(second)
        )
        topology_score = _negative_squared_distance(topology_first, topology_second)

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

        method_scores = {
            "mds": mds_score,
            "topology": topology_score,
            "gsgm": gsgm_score,
            "joint": joint_score,
        }
        metrics_for_pair = {}
        for method, score in method_scores.items():
            metrics, predictions = _evaluate_score(
                score, first, second, seeds
            )
            metrics_for_pair[method] = metrics
            totals[method]["trials"] += metrics["trials"]
            totals[method]["hit_sum"] += metrics["hit_at_1"] * metrics["trials"]
            totals[method]["mrr_sum"] += metrics["mrr"] * metrics["trials"]
            all_predictions[(first_index, second_index, method)] = predictions
        pair_rows.append(
            {
                "first_shard": first_index,
                "second_shard": second_index,
                "overlap": overlap_count,
                "declared_seeds": seed_count,
                "metrics": metrics_for_pair,
            }
        )

    aggregate = {}
    for method, values in totals.items():
        trials = values["trials"]
        aggregate[method] = {
            "trials": int(trials),
            "hit_at_1": 0.0 if trials == 0 else values["hit_sum"] / trials,
            "mrr": 0.0 if trials == 0 else values["mrr_sum"] / trials,
        }

    cycle_success = 0
    cycle_trials = 0
    selected_shards_set = sorted(
        set(
            value
            for first, second, _ in pairs
            for value in (first, second)
        )
    )
    for first in selected_shards_set:
        for second in selected_shards_set:
            if second <= first:
                continue
            for third in selected_shards_set:
                if third <= second:
                    continue
                first_second = all_predictions.get((first, second, "joint"))
                second_third = all_predictions.get((second, third, "joint"))
                first_third = all_predictions.get((first, third, "joint"))
                if first_second is None or second_third is None or first_third is None:
                    continue
                for source, middle in first_second.items():
                    if middle not in second_third or source not in first_third:
                        continue
                    cycle_trials += 1
                    if second_third[middle] == first_third[source]:
                        cycle_success += 1

    return {
        "scope": (
            "deterministic sampled k-NN graph inference; metrics are empirical "
            "and are not converted into a mechanism-level XDP budget"
        ),
        "background_records": int(vectors.shape[0]),
        "graph_pairs_evaluated": len(pair_rows),
        "aggregate": aggregate,
        "joint_cycle_consistency": (
            None if cycle_trials == 0 else cycle_success / float(cycle_trials)
        ),
        "joint_cycle_trials": cycle_trials,
        "pair_details": pair_rows,
    }
