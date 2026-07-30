import math
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.attack_channel import (  # noqa: E402
    METHODS,
    _attack_profile,
    _linked_replica_xdp_exposure,
    _rr_log_likelihood_ratio,
    evaluate_attack_channels,
)
from mess_xdp.data import ChallengePair, IsoHashModel  # noqa: E402
from mess_xdp.xdp import rr_epsilon_bit  # noqa: E402


def small_model(shards=4, bits=8):
    dimension = 2
    weights = []
    thresholds = []
    for shard in range(shards):
        angles = (
            np.arange(bits, dtype=np.float64) / bits * math.pi
            + shard * 0.09
        )
        weights.append(
            np.vstack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        )
        thresholds.append(np.zeros(bits, dtype=np.float32))
    return IsoHashModel(
        mean=np.zeros(dimension, dtype=np.float32),
        weights=weights,
        thresholds=thresholds,
        source_bits=shards * bits,
        requested_total_bits=shards * bits,
        tiled_repetitions=1,
        has_distinct_shard_blocks=True,
        layout="synthetic",
    )


class AttackChannelTests(unittest.TestCase):
    def test_rr_likelihood_ratio_uses_only_distinguishing_coordinates(self):
        eta = rr_epsilon_bit(0.2)
        clean0 = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        clean1 = np.asarray([1, 0, 0, 1], dtype=np.uint8)
        report = np.asarray([0, 1, 0, 0], dtype=np.uint8)
        # First differing coordinate supports world 0; the third supports
        # world 1.  Equal coordinates cancel.
        self.assertAlmostEqual(
            _rr_log_likelihood_ratio(report, clean0, clean1, eta),
            0.0,
        )

    def test_profile_uses_independent_calibration_edges(self):
        result = _attack_profile(
            calibration_world0=[-2, -1, 0, 1, 2, 3],
            calibration_world1=[-3, -2, -1, 0, 1, 2],
            test_world0=[0, 1, 1, 2, 2, 3, 3, 4],
            test_world1=[-2, -1, 0, 0, 1, 1, 2, 2],
            requested_bins=3,
            deltas=[0.1],
            profile_beta=0.05,
        )
        self.assertGreaterEqual(result["bins"], 2)
        self.assertEqual(len(result["delta_rows"]), 1)
        self.assertIn(
            result["delta_rows"][0]["confidence_status"],
            ("resolved", "insufficient_trials"),
        )

    def test_linked_replica_xdp_starts_from_single_shard_baseline(self):
        result = _linked_replica_xdp_exposure(
            replicas_world0=[1] * 16 + [2] * 4,
            replicas_world1=[1] * 15 + [2] * 5,
            single_shard_xdp=20.0,
            single_shard_delta=0.01,
            tail_probabilities=[0.2],
            confidence_beta_per_limit=0.05,
        )
        self.assertGreaterEqual(result["mean_xdp_exposure"], 20.0)
        row = result["tail_rows"][0]
        self.assertEqual(row["status"], "resolved")
        self.assertEqual(row["replica_count_upper_limit"], 2)
        self.assertEqual(row["xdp_upper_limit"], 40.0)
        self.assertEqual(row["basic_composition_delta"], 0.02)

    def test_two_world_audit_runs_without_target_route_labels(self):
        angles = np.linspace(0.0, 1.2, 48)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(
            np.float32
        )
        pair = ChallengePair(
            first_id=0,
            second_id=1,
            group="synthetic_near",
            target_angular_distance=0.05,
            angular_distance=float((angles[1] - angles[0]) / math.pi),
        )
        result = evaluate_attack_channels(
            vectors=vectors,
            model=small_model(),
            challenge_pairs=[pair],
            background_records=32,
            calibration_runs_per_world=2,
            test_runs_per_world=3,
            selected_shards=3,
            flip_probability=0.21,
            neighbors=4,
            seed_fraction=0.25,
            mds_dimensions=2,
            graph_signal_hops=2,
            aggregation_sizes=[0, 1],
            known_target_anchors=2,
            single_shard_xdp=20.0,
            single_shard_delta=0.01,
            exposure_tail_probabilities=[0.1],
            score_bins=2,
            deltas=[0.1],
            angular_radii=[0.05],
            confidence_beta=0.05,
            seed=77,
        )
        methods = result["pairs"][0]["methods"]
        self.assertEqual(set(methods), set(METHODS))
        for row in methods.values():
            self.assertGreaterEqual(row["mean_true_replicas_recovered"], 2.0)
            self.assertLessEqual(row["mean_true_replicas_recovered"], 3.0)
            self.assertIn(row["chosen_additional_candidates"], (0, 1))
            self.assertIn("privacy_profile", row)
            exposure = row["linked_replica_xdp_exposure"]
            self.assertGreaterEqual(exposure["mean_xdp_exposure"], 40.0)
            self.assertIn(
                exposure["tail_rows"][0]["status"],
                ("resolved", "insufficient_samples"),
            )
            self.assertIn("linkage_aware_xdp", row)
            self.assertIn("reconstructed_vector_xdp", row)
            self.assertGreaterEqual(
                row["linkage_aware_xdp"]["mean_effective_distance"],
                0.0,
            )
        self.assertEqual(result["known_target_anchors"], 2)
        self.assertGreaterEqual(
            result["linked_replica_xdp_global"]["max_mean_xdp_exposure"],
            40.0,
        )
        self.assertIn("reconstructed_angular_xdp_curve", result)


if __name__ == "__main__":
    unittest.main()
