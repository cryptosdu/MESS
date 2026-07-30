import math
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.angular_envelope import (  # noqa: E402
    empirical_angular_envelope,
    one_sided_tolerance_limit,
    prepare_angular_pool,
    required_samples_for_maximum,
    sample_pairs_within_radius,
)
from mess_xdp.data import IsoHashModel  # noqa: E402


def small_model(shards=4, bits=3):
    dimension = 2
    weights = []
    thresholds = []
    for shard in range(shards):
        angles = (
            np.arange(bits, dtype=np.float64) / bits * math.pi
            + shard * 0.07
        )
        weights.append(
            np.vstack([np.cos(angles), np.sin(angles)]).astype(
                np.float32
            )
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


class AngularEnvelopeTests(unittest.TestCase):
    def test_required_samples(self):
        self.assertEqual(required_samples_for_maximum(0.01, 0.05), 299)

    def test_insufficient_samples_are_not_given_a_finite_bound(self):
        limit = one_sided_tolerance_limit(
            [7] * 100, tail_probability=0.01, confidence_beta=0.05
        )
        self.assertFalse(limit.sufficient_samples)
        self.assertIsNone(limit.upper_limit)

    def test_sufficient_order_statistic(self):
        limit = one_sided_tolerance_limit(
            [7] * 400, tail_probability=0.01, confidence_beta=0.05
        )
        self.assertTrue(limit.sufficient_samples)
        self.assertEqual(limit.upper_limit, 7.0)

    def test_sampled_pairs_respect_the_angular_radius(self):
        angles = np.linspace(0.0, 0.8, 80)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(
            np.float32
        )
        pool = prepare_angular_pool(
            vectors=vectors,
            maximum_radius=0.08,
            pool_size=80,
            anchor_count=40,
            seed=12,
        )
        first, second, distances, eligible = sample_pairs_within_radius(
            pool=pool,
            radius=0.05,
            trials=100,
            seed=13,
        )
        self.assertEqual(first.size, 100)
        self.assertEqual(second.size, 100)
        self.assertGreater(eligible, 0)
        self.assertTrue(np.all(distances <= 0.05 + 1e-7))

    def test_complete_envelope_reports_single_and_complete_views(self):
        angles = np.linspace(0.0, 0.8, 80)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(
            np.float32
        )
        result = empirical_angular_envelope(
            vectors=vectors,
            model=small_model(),
            radii=[0.03, 0.05],
            trials_per_radius=600,
            pool_size=80,
            anchor_count=40,
            selected_shards=2,
            flip_probability=0.21,
            tail_probabilities=[0.01],
            simultaneous_beta=0.05,
            seed=19,
        )
        views = result["radii"][0]["views"]
        self.assertEqual(
            [view["observed_shards"] for view in views], [1, 2]
        )
        for view in views:
            tail = view["tail_bounds"][0]
            self.assertTrue(
                tail["direct_privacy_loss_tail"]["sufficient_samples"]
            )
            self.assertIsNotNone(
                tail["direct_privacy_loss_tail"][
                    "privacy_loss_epsilon_upper"
                ]
            )
        for view_index in range(2):
            budgets = [
                radius["views"][view_index]["tail_bounds"][0][
                    "direct_privacy_loss_tail"
                ]["privacy_loss_epsilon_upper"]
                for radius in result["radii"]
            ]
            self.assertEqual(budgets, sorted(budgets))


if __name__ == "__main__":
    unittest.main()
