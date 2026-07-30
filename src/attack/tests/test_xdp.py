import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.xdp import (  # noqa: E402
    original_lshrr_xdp,
    paper_regression_rows,
    rr_epsilon_bit,
)


class XDPTests(unittest.TestCase):
    def test_required_paper_baseline(self):
        result = original_lshrr_xdp(
            flip_probability=0.21,
            hash_bits=128,
            input_distance=0.05,
            target_delta=0.01,
        )
        self.assertAlmostEqual(result.epsilon_bit, 1.3249254147435987, places=12)
        self.assertAlmostEqual(result.alpha, 0.06817338216011763, places=12)
        self.assertAlmostEqual(result.xdp_xi_proposition5, 20.041077423378898, places=10)
        self.assertAlmostEqual(result.ordinary_ldp_epsilon, 169.59045308718063, places=10)

    def test_delta_is_not_angular_distance(self):
        result = original_lshrr_xdp(0.21, 128, 0.01, 0.01)
        self.assertAlmostEqual(result.xdp_xi_proposition5, 7.921694679669439, places=10)

    def test_paper_table(self):
        rows = paper_regression_rows()
        self.assertTrue(rows)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_rr_parameter(self):
        self.assertAlmostEqual(rr_epsilon_bit(0.21), math.log(0.79 / 0.21))


if __name__ == "__main__":
    unittest.main()

