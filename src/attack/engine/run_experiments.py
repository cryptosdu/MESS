#!/usr/bin/env python3
"""Run the complete MESS XDP and empirical leakage evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from mess_xdp import __version__
from mess_xdp.attack_channel import evaluate_attack_channels
from mess_xdp.angular_envelope import empirical_angular_envelope
from mess_xdp.data import (
    DATASETS,
    ChallengePair,
    discover_datasets,
    load_isohash,
    load_vectors,
    pair_shard_distances,
    resolve_dataset,
    select_angular_pairs,
    file_identity,
)
from mess_xdp.linkage import evaluate_linkage
from mess_xdp.pld import account_fixed_mapping_pair
from mess_xdp.query import query_privacy_rows
from mess_xdp.xdp import (
    original_lshrr_xdp,
    original_multigraph_reference,
    paper_regression_rows,
    rr_epsilon_bit,
)


PROFILES = {
    "quick": {
        "max_load_vectors": 800,
        "pair_pool_size": 500,
        "pair_anchors": 128,
        "pairs_per_target": 1,
        "angular_envelope_pool_size": 800,
        "angular_envelope_anchors": 256,
        "angular_envelope_trials": 600,
        "linkage_records": 120,
        "linkage_graph_pairs": 2,
        "attack_audit_background": 64,
        "attack_audit_pairs": 1,
        "attack_calibration_runs_per_world": 4,
        "attack_test_runs_per_world": 12,
    },
    "standard": {
        "max_load_vectors": 5000,
        "pair_pool_size": 3000,
        "pair_anchors": 512,
        "pairs_per_target": 3,
        "angular_envelope_pool_size": 5000,
        "angular_envelope_anchors": 1024,
        "angular_envelope_trials": 1000,
        "linkage_records": 400,
        "linkage_graph_pairs": 8,
        "attack_audit_background": 96,
        "attack_audit_pairs": 2,
        "attack_calibration_runs_per_world": 20,
        "attack_test_runs_per_world": 120,
    },
    "full": {
        "max_load_vectors": 20000,
        "pair_pool_size": 10000,
        "pair_anchors": 1500,
        "pairs_per_target": 10,
        "angular_envelope_pool_size": 20000,
        "angular_envelope_anchors": 3000,
        "angular_envelope_trials": 5000,
        "linkage_records": 1000,
        "linkage_graph_pairs": 20,
        "attack_audit_background": 160,
        "attack_audit_pairs": 3,
        "attack_calibration_runs_per_world": 30,
        "attack_test_runs_per_world": 300,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MESS XDP evaluation: original-paper regression, actual fixed-IsoHash "
            "multi-shard accounting, an empirical angular-distance envelope, "
            "PRR/IRR query accounting, deterministic cross-shard inference, "
            "and a held-out two-world attack-channel privacy profile."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS) + ["all"],
        default="sift",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument(
        "--mode",
        choices=["all", "xdp", "query", "inference", "attack", "validation"],
        default="all",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--strict-all", action="store_true")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--dataset-file", default="")
    parser.add_argument("--weight-file", default="")
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--selected-shards", type=int, default=16)
    parser.add_argument("--bits-per-shard", type=int, default=128)
    parser.add_argument("--flip-probability", type=float, default=0.08)
    parser.add_argument("--normalize-centered", action="store_true")
    parser.add_argument("--deltas", type=float, nargs="+", default=[1e-2, 1e-3, 1e-5])
    parser.add_argument(
        "--target-angular-distances",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
    )
    parser.add_argument(
        "--reference-flip-probability", type=float, default=0.21
    )
    parser.add_argument(
        "--reference-angular-distance", type=float, default=0.05
    )
    parser.add_argument("--reference-delta", type=float, default=0.01)

    parser.add_argument("--max-load-vectors", type=int, default=0)
    parser.add_argument("--pair-pool-size", type=int, default=0)
    parser.add_argument("--pair-anchors", type=int, default=0)
    parser.add_argument("--pairs-per-target", type=int, default=0)
    parser.add_argument("--angular-envelope-pool-size", type=int, default=0)
    parser.add_argument("--angular-envelope-anchors", type=int, default=0)
    parser.add_argument("--angular-envelope-trials", type=int, default=0)
    parser.add_argument(
        "--angular-tail-probabilities",
        type=float,
        nargs="+",
        default=[1e-2],
        help=(
            "Tail probabilities for the empirical angular envelope. Smaller "
            "values require substantially more trials."
        ),
    )
    parser.add_argument(
        "--angular-confidence-beta",
        type=float,
        default=0.05,
        help="Simultaneous statistical failure probability (default: 0.05).",
    )
    parser.add_argument(
        "--skip-angular-envelope",
        action="store_true",
        help="Skip the fixed-IsoHash angular-distance sampling experiment.",
    )
    parser.add_argument("--seed", type=int, default=20260724)

    parser.add_argument("--prr-flip", type=float, default=0.25)
    parser.add_argument("--irr-flip", type=float, default=0.20)
    parser.add_argument("--query-repeats", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument(
        "--query-contacted-shards",
        type=int,
        default=0,
        help="0 means that the query report is produced for every shard.",
    )

    parser.add_argument("--linkage-records", type=int, default=0)
    parser.add_argument("--linkage-graph-pairs", type=int, default=0)
    parser.add_argument("--linkage-knn-k", type=int, default=10)
    parser.add_argument("--linkage-seed-fraction", type=float, default=0.10)
    parser.add_argument("--mds-dimensions", type=int, default=8)
    parser.add_argument("--graph-signal-hops", type=int, default=3)
    parser.add_argument(
        "--skip-attack-channel",
        action="store_true",
        help=(
            "Skip the two-world leakage evaluation that connects matching "
            "outputs to an empirical attack-channel privacy profile."
        ),
    )
    parser.add_argument("--attack-audit-background", type=int, default=0)
    parser.add_argument("--attack-audit-pairs", type=int, default=0)
    parser.add_argument(
        "--attack-calibration-runs-per-world",
        type=int,
        default=0,
    )
    parser.add_argument("--attack-test-runs-per-world", type=int, default=0)
    parser.add_argument("--attack-score-bins", type=int, default=5)
    parser.add_argument("--attack-confidence-beta", type=float, default=0.05)
    parser.add_argument(
        "--attack-seed-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of common background correspondences supplied to the "
            "two-world attack channel (default: 0.20)."
        ),
    )
    parser.add_argument(
        "--attack-known-anchors",
        type=int,
        default=1,
        help=(
            "Number of target occurrences supplied as auxiliary anchors; "
            "the remaining route and replicas stay hidden (default: 1)."
        ),
    )
    parser.add_argument(
        "--single-shard-xdp",
        type=float,
        default=20.0,
        help=(
            "Declared isolated single-shard XDP baseline used to convert "
            "correctly linked replica counts into multi-shard exposure "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--single-shard-delta",
        type=float,
        default=0.01,
        help=(
            "Delta paired with --single-shard-xdp for basic composition "
            "(default: 0.01)."
        ),
    )
    parser.add_argument(
        "--attack-exposure-tail-probabilities",
        type=float,
        nargs="+",
        default=[0.05],
        help=(
            "Tail probabilities for high-probability linked-replica XDP "
            "exposure statements."
        ),
    )
    parser.add_argument(
        "--attack-aggregation-sizes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 15],
        help=(
            "Declared numbers of additional ranked candidates considered on "
            "calibration trials; zero is the anchor-only fallback."
        ),
    )
    parser.add_argument(
        "--attack-audit-angular-radius",
        type=float,
        default=0.10,
        help=(
            "Maximum angular radius admitted to the reconstruction audit "
            "(default: 0.10)."
        ),
    )
    parser.add_argument(
        "--reconstruction-xdp-angular-radii",
        type=float,
        nargs="+",
        default=[0.05, 0.10],
        help=(
            "Angular radii at which the reconstructed multi-shard XDP curve "
            "is reported (default: 0.05 0.10)."
        ),
    )
    return parser


def profile_value(args: argparse.Namespace, name: str) -> int:
    explicit = int(getattr(args, name))
    return explicit if explicit > 0 else int(PROFILES[args.profile][name])


def validate_arguments(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 1 <= args.selected_shards <= args.num_shards:
        raise ValueError("--selected-shards must lie in [1,num-shards]")
    if args.bits_per_shard < 1:
        raise ValueError("--bits-per-shard must be positive")
    if not 0.0 < args.flip_probability < 0.5:
        raise ValueError("--flip-probability must lie in (0,0.5)")
    if not 0.0 < args.reference_flip_probability < 0.5:
        raise ValueError("--reference-flip-probability must lie in (0,0.5)")
    if any(not 0.0 < value < 1.0 for value in args.deltas):
        raise ValueError("all --deltas must lie in (0,1)")
    if any(not 0.0 < value <= 1.0 for value in args.target_angular_distances):
        raise ValueError("all angular distances must lie in (0,1]")
    if any(not 0.0 < value < 1.0 for value in args.angular_tail_probabilities):
        raise ValueError("all --angular-tail-probabilities must lie in (0,1)")
    if not 0.0 < args.angular_confidence_beta < 1.0:
        raise ValueError("--angular-confidence-beta must lie in (0,1)")
    if args.dataset == "all" and (args.dataset_file or args.weight_file):
        raise ValueError("--dataset-file/--weight-file cannot be shared by --dataset all")
    if args.attack_score_bins < 2:
        raise ValueError("--attack-score-bins must be at least two")
    if not 0.0 < args.attack_confidence_beta < 1.0:
        raise ValueError("--attack-confidence-beta must lie in (0,1)")
    if not 0.0 < args.attack_seed_fraction <= 1.0:
        raise ValueError("--attack-seed-fraction must lie in (0,1]")
    if not 1 <= args.attack_known_anchors <= args.selected_shards:
        raise ValueError(
            "--attack-known-anchors must lie in [1,selected-shards]"
        )
    if args.single_shard_xdp < 0.0:
        raise ValueError("--single-shard-xdp cannot be negative")
    if not 0.0 <= args.single_shard_delta < 1.0:
        raise ValueError("--single-shard-delta must lie in [0,1)")
    if any(
        not 0.0 < value < 1.0
        for value in args.attack_exposure_tail_probabilities
    ):
        raise ValueError(
            "all --attack-exposure-tail-probabilities must lie in (0,1)"
        )
    if any(value < 0 for value in args.attack_aggregation_sizes):
        raise ValueError("--attack-aggregation-sizes cannot contain negatives")
    if not 0.0 < args.attack_audit_angular_radius <= 1.0:
        raise ValueError("--attack-audit-angular-radius must lie in (0,1]")
    if any(
        not 0.0 < value <= args.attack_audit_angular_radius
        for value in args.reconstruction_xdp_angular_radii
    ):
        raise ValueError(
            "all --reconstruction-xdp-angular-radii must lie in "
            "(0,--attack-audit-angular-radius]"
        )


def _reference_results(args: argparse.Namespace) -> Dict[str, object]:
    baseline = original_lshrr_xdp(
        flip_probability=args.reference_flip_probability,
        hash_bits=128,
        input_distance=args.reference_angular_distance,
        target_delta=args.reference_delta,
    ).to_dict()
    baseline["regression_target"] = (
        "p=0.21,kappa=128,d_theta=0.05,delta=0.01 should give xi about 20.04"
    )

    configured = []
    for distance in args.target_angular_distances:
        for delta in args.deltas:
            single = original_lshrr_xdp(
                flip_probability=args.flip_probability,
                hash_bits=args.bits_per_shard,
                input_distance=distance,
                target_delta=delta,
            ).to_dict()
            single["observed_shards"] = 1
            single["bits_per_shard"] = args.bits_per_shard
            configured.append(single)
            multi = original_multigraph_reference(
                flip_probability=args.flip_probability,
                bits_per_shard=args.bits_per_shard,
                shard_counts=[args.selected_shards],
                input_distance=distance,
                target_delta=delta,
            )[0]
            configured.append(multi)
    return {
        "paper_baseline": baseline,
        "paper_table_1_regression": paper_regression_rows(),
        "configured_random_lsh_reference": configured,
        "applicability": (
            "Reference only for independent random LSH bits satisfying the collision "
            "identity. It is not automatically claimed for pretrained IsoHash."
        ),
    }


def _pair_result(
    pair: ChallengePair,
    vectors: np.ndarray,
    model,
    args: argparse.Namespace,
) -> Dict[str, object]:
    distances = pair_shard_distances(
        vectors[pair.first_id],
        vectors[pair.second_id],
        model,
        normalize_centered=args.normalize_centered,
    )
    accounting = account_fixed_mapping_pair(
        shard_distances=distances,
        selected_shards=args.selected_shards,
        flip_probability=args.flip_probability,
        deltas=args.deltas,
    )
    query_shards = (
        args.num_shards
        if args.query_contacted_shards <= 0
        else int(args.query_contacted_shards)
    )
    if not 1 <= query_shards <= args.num_shards:
        raise ValueError("--query-contacted-shards must lie in [1,num-shards]")
    query_distances = distances[:query_shards]
    query_rows = query_privacy_rows(
        differing_coordinates=sum(query_distances),
        total_coordinates=query_shards * args.bits_per_shard,
        repeats=args.query_repeats,
        permanent_flip=args.prr_flip,
        instantaneous_flip=args.irr_flip,
    )
    result = pair.to_dict()
    result.update(
        {
            "shard_hamming_distances": distances,
            "all_shards_mismatch_rate": (
                sum(distances) / float(args.num_shards * args.bits_per_shard)
            ),
            "fixed_isohash_route_accounting": accounting.to_dict(),
            "query_accounting": query_rows,
        }
    )
    return result


def _aggregate_pairs(
    pairs: Sequence[Dict[str, object]], deltas: Sequence[float]
) -> Dict[str, object]:
    by_group = {}
    for pair in pairs:
        by_group.setdefault(pair["group"], []).append(pair)

    def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        summary = {
            "pairs": len(rows),
            "mean_angular_distance": float(
                np.mean([float(row["angular_distance"]) for row in rows])
            ),
            "minimum_angular_distance": float(
                min(float(row["angular_distance"]) for row in rows)
            ),
            "maximum_angular_distance": float(
                max(float(row["angular_distance"]) for row in rows)
            ),
            "mean_isohash_mismatch_rate": float(
                np.mean([float(row["all_shards_mismatch_rate"]) for row in rows])
            ),
            "delta_rows": [],
        }
        for index, delta in enumerate(deltas):
            route_values = [
                row["fixed_isohash_route_accounting"]["route_pxdp"][index]["xdp_xi"]
                for row in rows
            ]
            pld_values = [
                row["fixed_isohash_route_accounting"]["pld_approximate_dp"][index][
                    "epsilon"
                ]
                for row in rows
            ]
            summary["delta_rows"].append(
                {
                    "delta": float(delta),
                    "maximum_route_pxdp_xi": float(max(route_values)),
                    "median_route_pxdp_xi": float(np.median(route_values)),
                    "maximum_pld_epsilon": float(max(pld_values)),
                    "median_pld_epsilon": float(np.median(pld_values)),
                }
            )
        return summary

    return {
        "scope": (
            "finite evaluated pair set; values are not an unrestricted-domain maximum"
        ),
        "overall": summarize(pairs),
        "by_group": {group: summarize(rows) for group, rows in by_group.items()},
    }


def _flatten_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    baseline = result["original_xdp_reference"]["paper_baseline"]
    rows.append(
        {
            "category": "paper_baseline",
            "dataset": result["dataset"],
            "group": "d_theta_%.6g" % baseline["angular_distance"],
            "delta": baseline["target_delta"],
            "metric": "xdp_xi_proposition5",
            "value": baseline["xdp_xi_proposition5"],
        }
    )
    for reference in result["original_xdp_reference"][
        "configured_random_lsh_reference"
    ]:
        for metric in (
            "xdp_xi_proposition5",
            "xdp_xi_exact_binomial",
            "ordinary_ldp_epsilon",
        ):
            rows.append(
                {
                    "category": "random_lsh_reference",
                    "dataset": result["dataset"],
                    "angular_radius": reference["angular_distance"],
                    "observed_shards": reference["observed_shards"],
                    "tail_probability": reference["target_delta"],
                    "metric": metric,
                    "value": reference[metric],
                    "status": "reference_assumptions_required",
                }
            )
    for pair_index, pair in enumerate(result.get("challenge_pairs", [])):
        accounting = pair["fixed_isohash_route_accounting"]
        for row in accounting["route_pxdp"]:
            rows.append(
                {
                    "category": "fixed_isohash_pair",
                    "dataset": result["dataset"],
                    "pair": pair_index,
                    "group": pair["group"],
                    "angular_distance": pair["angular_distance"],
                    "delta": row["delta"],
                    "metric": "route_pxdp_xi",
                    "value": row["xdp_xi"],
                }
            )
        for row in accounting["pld_approximate_dp"]:
            rows.append(
                {
                    "category": "fixed_isohash_pair",
                    "dataset": result["dataset"],
                    "pair": pair_index,
                    "group": pair["group"],
                    "angular_distance": pair["angular_distance"],
                    "delta": row["delta"],
                    "metric": "pld_epsilon",
                    "value": row["epsilon"],
                }
            )

    envelope = result.get("empirical_angular_xdp_envelope")
    if envelope:
        for radius_row in envelope["radii"]:
            for view in radius_row.get("views", []):
                for tail in view["tail_bounds"]:
                    mismatch = tail["pure_loss_from_mismatch"]
                    loss = tail["direct_privacy_loss_tail"]
                    common = {
                        "category": "empirical_angular_xdp_envelope",
                        "dataset": result["dataset"],
                        "angular_radius": radius_row["angular_radius"],
                        "observed_shards": view["observed_shards"],
                        "view": view["view"],
                        "tail_probability": tail["tail_probability"],
                        "simultaneous_confidence": envelope[
                            "simultaneous_confidence"
                        ],
                        "eligible_anchors": radius_row["eligible_anchors"],
                        "trials": radius_row["trials"],
                    }
                    mismatch_row = dict(common)
                    mismatch_row.update(
                        {
                            "metric": "pure_xdp_xi_upper",
                            "value": mismatch["pure_xdp_xi_upper"],
                            "raw_value": mismatch.get(
                                "raw_pure_xdp_xi_upper"
                            ),
                            "status": mismatch["status"],
                            "required_samples": mismatch[
                                "required_samples_for_maximum"
                            ],
                        }
                    )
                    rows.append(mismatch_row)
                    loss_row = dict(common)
                    loss_row.update(
                        {
                            "metric": "privacy_loss_epsilon_upper",
                            "value": loss["privacy_loss_epsilon_upper"],
                            "raw_value": loss.get(
                                "raw_privacy_loss_epsilon_upper"
                            ),
                            "status": loss["status"],
                            "required_samples": loss[
                                "required_samples_for_maximum"
                            ],
                        }
                    )
                    rows.append(loss_row)

    inference = result.get("cross_shard_inference")
    if inference:
        for method, metrics in inference["aggregate"].items():
            for metric in ("hit_at_1", "mrr"):
                rows.append(
                    {
                        "category": "cross_shard_inference",
                        "dataset": result["dataset"],
                        "group": method,
                        "metric": metric,
                        "value": metrics[metric],
                    }
                )
    attack_channels = result.get("attack_channel_leakage")
    if attack_channels:
        for pair in attack_channels["pairs"]:
            for method, metrics in pair["methods"].items():
                common = {
                    "category": "attack_channel_leakage",
                    "dataset": result["dataset"],
                    "pair": pair["pair_index"],
                    "group": method,
                    "angular_distance": pair["angular_distance"],
                    "chosen_additional_candidates": metrics[
                        "chosen_additional_candidates"
                    ],
                }
                for metric in (
                    "auc_world0_over_world1",
                    "auc_best_direction",
                    "threshold_advantage",
                    "mean_true_replicas_recovered",
                ):
                    row = dict(common)
                    row.update({"metric": metric, "value": metrics[metric]})
                    rows.append(row)
                exposure = metrics["linked_replica_xdp_exposure"]
                for metric in (
                    "mean_xdp_exposure",
                    "median_xdp_exposure",
                ):
                    row = dict(common)
                    row.update({"metric": metric, "value": exposure[metric]})
                    rows.append(row)
                for tail_row in exposure["tail_rows"]:
                    for metric in (
                        "replica_count_upper_limit",
                        "xdp_upper_limit",
                        "basic_composition_delta",
                    ):
                        row = dict(common)
                        row.update(
                            {
                                "tail_probability": tail_row[
                                    "tail_probability"
                                ],
                                "metric": metric,
                                "value": tail_row[metric],
                                "status": tail_row["status"],
                            }
                        )
                        rows.append(row)
                for delta_row in metrics["privacy_profile"]["delta_rows"]:
                    for metric in (
                        "epsilon_point",
                        "epsilon_confidence_upper",
                    ):
                        row = dict(common)
                        row.update(
                            {
                                "delta": delta_row["delta"],
                                "metric": metric,
                                "value": delta_row[metric],
                                "status": delta_row["confidence_status"],
                            }
                        )
                        rows.append(row)
                linkage_xdp = metrics["linkage_aware_xdp"]
                null_check = metrics["failed_block_null_check"]
                rows.append(
                    {
                        **common,
                        "category": "linkage_aware_xdp",
                        "metric": "false_block_normalized_distance",
                        "value": null_check[
                            "observed_mean_normalized_distance_to_two_templates"
                        ],
                        "false_blocks": null_check["false_blocks_observed"],
                    }
                )
                for delta_row in linkage_xdp["delta_rows"]:
                    for metric in (
                        "xdp_total_budget_xi_point",
                        "xdp_per_unit_angular_distance_point",
                    ):
                        rows.append(
                            {
                                **common,
                                "category": "linkage_aware_xdp",
                                "delta": delta_row["delta"],
                                "metric": metric,
                                "value": delta_row[metric],
                            }
                        )
                for tail_row in linkage_xdp["tail_rows"]:
                    for metric in (
                        "effective_distance_upper_limit",
                        "xdp_pure_loss_upper_limit",
                    ):
                        rows.append(
                            {
                                **common,
                                "category": "linkage_aware_xdp",
                                "tail_probability": tail_row[
                                    "tail_probability"
                                ],
                                "metric": metric,
                                "value": tail_row[metric],
                                "status": tail_row["status"],
                            }
                        )
                for delta_row in metrics["reconstructed_vector_xdp"][
                    "delta_rows"
                ]:
                    for metric in (
                        "xdp_total_budget_xi_point",
                        "xdp_total_budget_xi_confidence",
                    ):
                        rows.append(
                            {
                                **common,
                                "category": "reconstructed_vector_xdp",
                                "delta": delta_row["delta"],
                                "metric": metric,
                                "value": delta_row[metric],
                                "status": delta_row["confidence_status"],
                            }
                        )
        for curve_row in attack_channels["reconstructed_angular_xdp_curve"][
            "rows"
        ]:
            for metric in (
                "linkage_xdp_xi_point_envelope",
                "score_xdp_xi_point_envelope",
                "score_xdp_xi_confidence_envelope",
            ):
                rows.append(
                    {
                        "category": "reconstructed_angular_xdp_curve",
                        "dataset": result["dataset"],
                        "group": curve_row["method"],
                        "angular_radius": curve_row["angular_radius"],
                        "delta": curve_row["delta"],
                        "evaluated_pairs": curve_row["evaluated_pairs"],
                        "metric": metric,
                        "value": curve_row[metric],
                        "status": curve_row["status"],
                    }
                )
        global_exposure = attack_channels["linked_replica_xdp_global"]
        rows.append(
            {
                "category": "linked_replica_xdp_global",
                "dataset": result["dataset"],
                "group": "all_methods_and_pairs",
                "metric": "max_mean_xdp_exposure",
                "value": global_exposure["max_mean_xdp_exposure"],
                "simultaneous_confidence": global_exposure[
                    "simultaneous_confidence"
                ],
            }
        )
        for tail_row in global_exposure["tail_rows"]:
            for metric in (
                "replica_count_upper_limit",
                "xdp_upper_limit",
                "basic_composition_delta",
            ):
                rows.append(
                    {
                        "category": "linked_replica_xdp_global",
                        "dataset": result["dataset"],
                        "group": "all_methods_and_pairs",
                        "tail_probability": tail_row[
                            "tail_probability"
                        ],
                        "metric": metric,
                        "value": tail_row[metric],
                        "status": tail_row["status"],
                        "simultaneous_confidence": global_exposure[
                            "simultaneous_confidence"
                        ],
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    keys = sorted(set(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_one(dataset: str, args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    started = time.time()
    preset, directory, dataset_path, weight_path = resolve_dataset(
        dataset=dataset,
        data_root=args.data_root,
        dataset_file=args.dataset_file,
        weight_file=args.weight_file,
    )
    maximum_vectors = profile_value(args, "max_load_vectors")
    print("[%s] loading up to %d vectors from %s" % (dataset, maximum_vectors, dataset_path))
    vectors = load_vectors(preset, dataset_path, maximum_vectors)
    model = load_isohash(
        weight_path,
        dimension=vectors.shape[1],
        num_shards=args.num_shards,
        bits_per_shard=args.bits_per_shard,
    )
    print("[%s] IsoHash: %s" % (dataset, model.layout))

    result = {
        "version": __version__,
        "dataset": dataset,
        "configuration": {
            "profile": args.profile,
            "mode": args.mode,
            "num_shards": args.num_shards,
            "selected_shards": args.selected_shards,
            "bits_per_shard": args.bits_per_shard,
            "flip_probability": args.flip_probability,
            "epsilon_bit": rr_epsilon_bit(args.flip_probability),
            "deltas": list(args.deltas),
            "target_angular_distances": list(args.target_angular_distances),
            "angular_tail_probabilities": list(
                args.angular_tail_probabilities
            ),
            "angular_confidence_beta": args.angular_confidence_beta,
            "attack_score_bins": args.attack_score_bins,
            "attack_confidence_beta": args.attack_confidence_beta,
            "attack_seed_fraction": args.attack_seed_fraction,
            "attack_known_anchors": args.attack_known_anchors,
            "single_shard_xdp": args.single_shard_xdp,
            "single_shard_delta": args.single_shard_delta,
            "attack_exposure_tail_probabilities": list(
                args.attack_exposure_tail_probabilities
            ),
            "attack_aggregation_sizes": list(
                args.attack_aggregation_sizes
            ),
            "attack_audit_angular_radius": args.attack_audit_angular_radius,
            "reconstruction_xdp_angular_radii": list(
                args.reconstruction_xdp_angular_radii
            ),
            "seed": args.seed,
        },
        "files": {
            "data_directory": str(directory),
            "dataset_file": str(dataset_path),
            "weight_file": str(weight_path),
            "dataset_identity": file_identity(dataset_path),
            "weight_identity": file_identity(weight_path),
        },
        "loaded_vectors": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "isohash": model.metadata(),
        "original_xdp_reference": _reference_results(args),
    }

    pairs = []
    if args.mode in (
        "all",
        "xdp",
        "query",
        "inference",
        "attack",
        "validation",
    ):
        pairs = select_angular_pairs(
            vectors=vectors,
            targets=args.target_angular_distances,
            pairs_per_target=profile_value(args, "pairs_per_target"),
            pool_size=profile_value(args, "pair_pool_size"),
            anchor_count=profile_value(args, "pair_anchors"),
            seed=args.seed,
            include_nearest=True,
        )
        pair_rows = [_pair_result(pair, vectors, model, args) for pair in pairs]
        result["challenge_pairs"] = pair_rows
        result["evaluated_set_summary"] = _aggregate_pairs(pair_rows, args.deltas)

    if (
        args.mode in ("all", "xdp", "validation")
        and not args.skip_angular_envelope
    ):
        print(
            "[%s] sampling fixed-IsoHash angular envelope (%d trials/radius)"
            % (dataset, profile_value(args, "angular_envelope_trials"))
        )
        result["empirical_angular_xdp_envelope"] = empirical_angular_envelope(
            vectors=vectors,
            model=model,
            radii=args.target_angular_distances,
            trials_per_radius=profile_value(
                args, "angular_envelope_trials"
            ),
            pool_size=profile_value(args, "angular_envelope_pool_size"),
            anchor_count=profile_value(args, "angular_envelope_anchors"),
            selected_shards=args.selected_shards,
            flip_probability=args.flip_probability,
            tail_probabilities=args.angular_tail_probabilities,
            simultaneous_beta=args.angular_confidence_beta,
            seed=args.seed + 424242,
            normalize_centered=args.normalize_centered,
            progress_callback=lambda message: print(
                "[%s] %s" % (dataset, message), flush=True
            ),
        )

    if args.mode in ("all", "inference", "attack"):
        linkage_records = min(profile_value(args, "linkage_records"), vectors.shape[0])
        linkage_rng = np.random.default_rng(args.seed + 9173)
        linkage_ids = linkage_rng.choice(
            vectors.shape[0], size=linkage_records, replace=False
        )
        result["cross_shard_inference"] = evaluate_linkage(
            vectors=np.asarray(vectors[linkage_ids], dtype=np.float32),
            model=model,
            selected_shards=args.selected_shards,
            flip_probability=args.flip_probability,
            neighbors=args.linkage_knn_k,
            seed_fraction=args.linkage_seed_fraction,
            maximum_graph_pairs=profile_value(args, "linkage_graph_pairs"),
            mds_dimensions=args.mds_dimensions,
            graph_signal_hops=args.graph_signal_hops,
            seed=args.seed,
            normalize_centered=args.normalize_centered,
        )

    if (
        args.mode in ("all", "inference", "attack")
        and not args.skip_attack_channel
    ):
        eligible_pairs = [
            pair
            for pair in pairs
            if float(pair.angular_distance)
            <= float(args.attack_audit_angular_radius) + 1e-12
        ]
        requested_pairs = profile_value(args, "attack_audit_pairs")
        audit_pairs = []
        used_pair_ids = set()
        for radius in sorted(
            set(float(value) for value in args.reconstruction_xdp_angular_radii)
        ):
            radius_candidates = [
                pair
                for pair in eligible_pairs
                if float(pair.angular_distance) <= radius + 1e-12
            ]
            radius_candidates.sort(
                key=lambda pair: (
                    abs(radius - float(pair.angular_distance)),
                    pair.first_id,
                    pair.second_id,
                )
            )
            added = 0
            for pair in radius_candidates:
                key = (int(pair.first_id), int(pair.second_id))
                if key in used_pair_ids:
                    continue
                used_pair_ids.add(key)
                audit_pairs.append(pair)
                added += 1
                if added >= requested_pairs:
                    break
        if not audit_pairs:
            raise RuntimeError(
                "no real challenge pairs satisfy the reconstruction XDP "
                "angular radii; increase the radii or pair-selection pool"
            )
        print(
            "[%s] running unlabelled two-world attack-channel audit "
            "(%d pair(s), %d anchor(s), %.0f%% auxiliary seeds, "
            "%d calibration + %d test runs/world)"
            % (
                dataset,
                len(audit_pairs),
                args.attack_known_anchors,
                100.0 * args.attack_seed_fraction,
                profile_value(args, "attack_calibration_runs_per_world"),
                profile_value(args, "attack_test_runs_per_world"),
            )
        )
        result["attack_channel_leakage"] = evaluate_attack_channels(
            vectors=vectors,
            model=model,
            challenge_pairs=audit_pairs,
            background_records=profile_value(
                args,
                "attack_audit_background",
            ),
            calibration_runs_per_world=profile_value(
                args,
                "attack_calibration_runs_per_world",
            ),
            test_runs_per_world=profile_value(
                args,
                "attack_test_runs_per_world",
            ),
            selected_shards=args.selected_shards,
            flip_probability=args.flip_probability,
            neighbors=args.linkage_knn_k,
            seed_fraction=args.attack_seed_fraction,
            mds_dimensions=args.mds_dimensions,
            graph_signal_hops=args.graph_signal_hops,
            aggregation_sizes=args.attack_aggregation_sizes,
            known_target_anchors=args.attack_known_anchors,
            single_shard_xdp=args.single_shard_xdp,
            single_shard_delta=args.single_shard_delta,
            exposure_tail_probabilities=(
                args.attack_exposure_tail_probabilities
            ),
            score_bins=args.attack_score_bins,
            deltas=args.deltas,
            angular_radii=args.reconstruction_xdp_angular_radii,
            confidence_beta=args.attack_confidence_beta,
            seed=args.seed + 818181,
            normalize_centered=args.normalize_centered,
            progress_callback=lambda message: print(
                "[%s] %s" % (dataset, message),
                flush=True,
            ),
        )

    result["elapsed_seconds"] = time.time() - started
    dataset_output = output_root / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    json_path = dataset_output / "security_results.json"
    csv_path = dataset_output / "security_results.csv"
    angular_csv_path = dataset_output / "angular_xdp_curve.csv"
    attack_csv_path = dataset_output / "attack_channel_leakage.csv"
    reconstruction_csv_path = (
        dataset_output / "reconstructed_multigraph_xdp.csv"
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    flat_rows = _flatten_rows(result)
    _write_csv(csv_path, flat_rows)
    angular_rows = [
        row
        for row in flat_rows
        if row.get("category") == "empirical_angular_xdp_envelope"
    ]
    if angular_rows:
        _write_csv(angular_csv_path, angular_rows)
    attack_rows = [
        row
        for row in flat_rows
        if row.get("category")
        in ("attack_channel_leakage", "linked_replica_xdp_global")
    ]
    if attack_rows:
        _write_csv(attack_csv_path, attack_rows)
    reconstruction_rows = [
        row
        for row in flat_rows
        if row.get("category")
        in (
            "linkage_aware_xdp",
            "reconstructed_vector_xdp",
            "reconstructed_angular_xdp_curve",
        )
    ]
    if reconstruction_rows:
        _write_csv(reconstruction_csv_path, reconstruction_rows)
    print("[%s] saved %s" % (dataset, json_path))
    print("[%s] saved %s" % (dataset, csv_path))
    if angular_rows:
        print("[%s] saved %s" % (dataset, angular_csv_path))
    if attack_rows:
        print("[%s] saved %s" % (dataset, attack_csv_path))
    if reconstruction_rows:
        print("[%s] saved %s" % (dataset, reconstruction_csv_path))
    return result


def print_summary(result: Dict[str, object]) -> None:
    baseline = result["original_xdp_reference"]["paper_baseline"]
    print(
        "[%s] paper regression: p=%.2f, k=%d, d_theta=%.2f, delta=%.2g -> xi=%.6f"
        % (
            result["dataset"],
            baseline["flip_probability"],
            baseline["hash_bits"],
            baseline["angular_distance"],
            baseline["target_delta"],
            baseline["xdp_xi_proposition5"],
        )
    )
    summary = result.get("evaluated_set_summary", {}).get("overall")
    if summary:
        print(
            "[%s] evaluated angular distance range: %.6f to %.6f"
            % (
                result["dataset"],
                summary["minimum_angular_distance"],
                summary["maximum_angular_distance"],
            )
        )
        for row in summary["delta_rows"]:
            print(
                "[%s] delta=%g: fixed-IsoHash route-PXDP xi max=%.6f; "
                "PLD epsilon max=%.6f"
                % (
                    result["dataset"],
                    row["delta"],
                    row["maximum_route_pxdp_xi"],
                    row["maximum_pld_epsilon"],
                )
            )

    envelope = result.get("empirical_angular_xdp_envelope")
    if envelope:
        confidence = envelope["simultaneous_confidence"]
        for radius_row in envelope["radii"]:
            if radius_row["status"] != "ok":
                print(
                    "[%s] angular radius<=%g: no eligible real pairs"
                    % (result["dataset"], radius_row["angular_radius"])
                )
                continue
            for view in radius_row["views"]:
                for tail in view["tail_bounds"]:
                    loss = tail["direct_privacy_loss_tail"]
                    pure = tail["pure_loss_from_mismatch"]
                    if loss["sufficient_samples"]:
                        print(
                            "[%s] d_theta<=%g, shards=%d: with %.1f%% "
                            "simultaneous confidence, Pr[L>%.6f]<=%g; "
                            "pure-loss envelope xi<=%.6f"
                            % (
                                result["dataset"],
                                radius_row["angular_radius"],
                                view["observed_shards"],
                                100.0 * confidence,
                                loss["privacy_loss_epsilon_upper"],
                                tail["tail_probability"],
                                pure["pure_xdp_xi_upper"],
                            )
                        )
                    else:
                        print(
                            "[%s] d_theta<=%g, shards=%d, tail=%g: "
                            "insufficient trials (%d; need at least %d)"
                            % (
                                result["dataset"],
                                radius_row["angular_radius"],
                                view["observed_shards"],
                                tail["tail_probability"],
                                loss["sample_count"],
                                loss["required_samples_for_maximum"],
                            )
                        )

    inference = result.get("cross_shard_inference")
    if inference:
        for method, metrics in inference["aggregate"].items():
            print(
                "[%s] inference %s: trials=%d, Hit@1=%.6f, MRR=%.6f"
                % (
                    result["dataset"],
                    method,
                    metrics["trials"],
                    metrics["hit_at_1"],
                    metrics["mrr"],
                )
            )

    attack_channels = result.get("attack_channel_leakage")
    if attack_channels:
        for pair in attack_channels["pairs"]:
            for method, metrics in pair["methods"].items():
                print(
                    "[%s] attack-channel pair=%d method=%s: AUC=%.6f, "
                    "advantage=%.6f, chosen extra=%d, "
                    "mean true replicas=%.3f, mean linked-XDP=%.3f"
                    % (
                        result["dataset"],
                        pair["pair_index"],
                        method,
                        metrics["auc_best_direction"],
                        metrics["threshold_advantage"],
                        metrics["chosen_additional_candidates"],
                        metrics["mean_true_replicas_recovered"],
                        metrics["linked_replica_xdp_exposure"][
                            "mean_xdp_exposure"
                        ],
                    )
                )
                for row in metrics["linked_replica_xdp_exposure"][
                    "tail_rows"
                ]:
                    print(
                        "[%s]   linked-XDP tail=%g: replicas<=%s, "
                        "xi<=%s, composed delta=%s (%s)"
                        % (
                            result["dataset"],
                            row["tail_probability"],
                            "unresolved"
                            if row["replica_count_upper_limit"] is None
                            else "%d" % row["replica_count_upper_limit"],
                            "unresolved"
                            if row["xdp_upper_limit"] is None
                            else "%.6f" % row["xdp_upper_limit"],
                            "unresolved"
                            if row["basic_composition_delta"] is None
                            else "%.6g" % row["basic_composition_delta"],
                            row["status"],
                        )
                    )
                linkage_xdp = metrics["linkage_aware_xdp"]
                null_check = metrics["failed_block_null_check"]
                for row in linkage_xdp["delta_rows"]:
                    print(
                        "[%s]   reconstructed XDP delta=%g: xi=%.6f "
                        "(effective hash distance mean=%.3f, "
                        "false-block null distance=%s)"
                        % (
                            result["dataset"],
                            row["delta"],
                            row["xdp_total_budget_xi_point"],
                            linkage_xdp["mean_effective_distance"],
                            "n/a"
                            if null_check[
                                "observed_mean_normalized_distance_to_two_templates"
                            ]
                            is None
                            else "%.6f"
                            % null_check[
                                "observed_mean_normalized_distance_to_two_templates"
                            ],
                        )
                    )
        global_exposure = attack_channels["linked_replica_xdp_global"]
        print(
            "[%s] global linked-replica XDP: max mean exposure=%.6f"
            % (
                result["dataset"],
                global_exposure["max_mean_xdp_exposure"],
            )
        )
        for row in global_exposure["tail_rows"]:
            print(
                "[%s]   global tail=%g: Pr[xi>%s]<=%g with %.1f%% "
                "simultaneous confidence (%s)"
                % (
                    result["dataset"],
                    row["tail_probability"],
                    "unresolved"
                    if row["xdp_upper_limit"] is None
                    else "%.6f" % row["xdp_upper_limit"],
                    row["tail_probability"],
                    100.0 * global_exposure["simultaneous_confidence"],
                    row["status"],
                )
            )
        for row in attack_channels["reconstructed_angular_xdp_curve"][
            "rows"
        ]:
            print(
                "[%s] reconstructed angular-XDP radius<=%g method=%s "
                "delta=%g: linkage xi=%s, end-to-end score xi=%s (%s)"
                % (
                    result["dataset"],
                    row["angular_radius"],
                    row["method"],
                    row["delta"],
                    "n/a"
                    if row["linkage_xdp_xi_point_envelope"] is None
                    else "%.6f"
                    % row["linkage_xdp_xi_point_envelope"],
                    "n/a"
                    if row["score_xdp_xi_point_envelope"] is None
                    else "%.6f"
                    % row["score_xdp_xi_point_envelope"],
                    row["status"],
                )
            )


def check_layout(args: argparse.Namespace) -> int:
    discovery = discover_datasets(args.data_root)
    print("MESS XDP package %s" % __version__)
    print("Dataset discovery:")
    for name in sorted(discovery):
        row = discovery[name]
        print(
            "  %-8s ready=%s\n    data=%s\n    weights=%s"
            % (name, row["ready"], row["dataset_file"], row["weight_file"])
        )
    if args.dataset == "all":
        unavailable = [
            name for name in sorted(DATASETS) if not discovery[name]["ready"]
        ]
        return 1 if unavailable else 0

    try:
        preset, _, dataset_path, weight_path = resolve_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            dataset_file=args.dataset_file,
            weight_file=args.weight_file,
        )
        sample = load_vectors(preset, dataset_path, 2)
        model = load_isohash(
            weight_path,
            dimension=sample.shape[1],
            num_shards=args.num_shards,
            bits_per_shard=args.bits_per_shard,
        )
        print("Selected dataset check: OK")
        print("  dimension=%d" % sample.shape[1])
        print("  IsoHash=%s" % model.layout)
        print("  distinct shard blocks=%s" % model.has_distinct_shard_blocks)
        return 0
    except Exception as error:
        print("Selected dataset check: FAILED: %s: %s" % (type(error).__name__, error))
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args)
    if args.check:
        return check_layout(args)

    package_root = Path(__file__).resolve().parent.parent
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else package_root / "results"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    failures = []
    for dataset in selected:
        try:
            result = run_one(dataset, args, output_root)
            results.append(result)
            print_summary(result)
        except Exception as error:
            failures.append({"dataset": dataset, "error": "%s: %s" % (type(error).__name__, error)})
            print("[%s] FAILED: %s: %s" % (dataset, type(error).__name__, error), file=sys.stderr)
            if args.dataset != "all" or args.strict_all:
                raise

    summary_directory = output_root / "all_datasets"
    summary_directory.mkdir(parents=True, exist_ok=True)
    summary_path = summary_directory / "security_summary.json"
    summary_csv_path = summary_directory / "security_summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": __version__,
                "succeeded": [result["dataset"] for result in results],
                "failed": failures,
                "paper_baseline_xi": (
                    None
                    if not results
                    else results[0]["original_xdp_reference"]["paper_baseline"][
                        "xdp_xi_proposition5"
                    ]
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    combined_rows = []
    for result in results:
        combined_rows.extend(_flatten_rows(result))
    if combined_rows:
        _write_csv(summary_csv_path, combined_rows)
    print("Saved cross-dataset summary: %s" % summary_path)
    if combined_rows:
        print("Saved cross-dataset table: %s" % summary_csv_path)
    return 1 if failures and (args.strict_all or not results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
