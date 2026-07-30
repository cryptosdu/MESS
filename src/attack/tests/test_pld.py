import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.pld import (  # noqa: E402
    account_fixed_mapping_pair,
    route_mixture_rr_pld,
    uniform_route_sum_distribution,
)


class PLDTests(unittest.TestCase):
    def test_route_distribution(self):
        distribution = uniform_route_sum_distribution([1, 2, 3], 2)
        self.assertEqual(set(distribution), {3, 4, 5})
        for value in distribution.values():
            self.assertAlmostEqual(value, 1.0 / 3.0)

    def test_zero_distance_has_zero_loss(self):
        pld = route_mixture_rr_pld({0: 1.0}, 0.21)
        self.assertEqual(pld.pure_epsilon, 0.0)
        self.assertEqual(pld.delta(0.0), 0.0)

    def test_pld_is_tighter_than_pure_bound(self):
        result = account_fixed_mapping_pair(
            shard_distances=[2, 3, 4, 5],
            selected_shards=2,
            flip_probability=0.21,
            deltas=[0.01],
        )
        pure = result.epsilon_bit * result.maximum_route_distance
        approximate = result.pld_approximate_dp[0]["epsilon"]
        self.assertLessEqual(approximate, pure)


if __name__ == "__main__":
    unittest.main()

