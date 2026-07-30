import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.query import (  # noqa: E402
    query_privacy_rows,
    repeated_report_eta,
    search_pattern_gamma,
)


class QueryTests(unittest.TestCase):
    def test_one_report_reduces_to_effective_flip(self):
        a = 0.25
        b = 0.20
        effective = a + b - 2.0 * a * b
        expected = math.log((1.0 - effective) / effective)
        self.assertAlmostEqual(repeated_report_eta(1, a, b), expected, places=12)

    def test_search_pattern_leaks_correlation_for_equal_bits(self):
        gamma = search_pattern_gamma(4, 0.25, 0.20, False)
        self.assertGreater(gamma, 0.0)

    def test_query_rows_include_access_and_search(self):
        rows = query_privacy_rows(3, 10, [1, 4], 0.25, 0.20)
        self.assertEqual(len(rows), 2)
        self.assertIn("access_pattern_pair_xdp_xi", rows[0])
        self.assertNotIn("search_pattern_pair_epsilon", rows[0])
        self.assertIn("search_pattern_pair_epsilon", rows[1])


if __name__ == "__main__":
    unittest.main()

