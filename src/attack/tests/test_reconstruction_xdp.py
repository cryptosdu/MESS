import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.reconstruction_xdp import (  # noqa: E402
    angular_xdp_envelope,
    linkage_aware_xdp,
    reconstructed_score_as_xdp,
)


class ReconstructionXDPTests(unittest.TestCase):
    def test_world_independent_failed_blocks_have_zero_effective_xdp(self):
        result = linkage_aware_xdp(
            effective_distances_world0=[0] * 400,
            effective_distances_world1=[0] * 400,
            flip_probability=0.21,
            angular_distance=0.05,
            deltas=[0.01],
            tail_probabilities=[0.01],
            confidence_beta_per_limit=0.05,
        )
        self.assertEqual(
            result["delta_rows"][0]["xdp_total_budget_xi_point"],
            0.0,
        )
        self.assertEqual(
            result["tail_rows"][0]["xdp_pure_loss_upper_limit"],
            0.0,
        )

    def test_more_reconstructed_differing_bits_increase_budget(self):
        small = linkage_aware_xdp(
            effective_distances_world0=[2] * 40,
            effective_distances_world1=[2] * 40,
            flip_probability=0.21,
            angular_distance=0.05,
            deltas=[0.01],
            tail_probabilities=[0.1],
            confidence_beta_per_limit=0.05,
        )
        large = linkage_aware_xdp(
            effective_distances_world0=[10] * 40,
            effective_distances_world1=[10] * 40,
            flip_probability=0.21,
            angular_distance=0.05,
            deltas=[0.01],
            tail_probabilities=[0.1],
            confidence_beta_per_limit=0.05,
        )
        self.assertGreater(
            large["delta_rows"][0]["xdp_total_budget_xi_point"],
            small["delta_rows"][0]["xdp_total_budget_xi_point"],
        )

    def test_score_profile_is_annotated_with_input_distance(self):
        profile = {
            "profile_confidence": 0.95,
            "delta_rows": [
                {
                    "delta": 0.01,
                    "epsilon_point": 2.0,
                    "epsilon_confidence_upper": 3.0,
                    "confidence_status": "resolved",
                }
            ],
        }
        result = reconstructed_score_as_xdp(profile, angular_distance=0.05)
        row = result["delta_rows"][0]
        self.assertAlmostEqual(
            row["xdp_per_unit_angular_distance_point"],
            40.0,
        )
        self.assertAlmostEqual(
            row["xdp_per_unit_angular_distance_confidence"],
            60.0,
        )

    def test_angular_curve_uses_only_pairs_inside_radius(self):
        def pair(distance, value):
            method = {
                "linkage_aware_xdp": {
                    "delta_rows": [
                        {"xdp_total_budget_xi_point": float(value)}
                    ]
                },
                "reconstructed_vector_xdp": {
                    "delta_rows": [
                        {
                            "xdp_total_budget_xi_point": float(value) / 2.0,
                            "xdp_total_budget_xi_confidence": float(value),
                        }
                    ]
                },
            }
            return {
                "angular_distance": float(distance),
                "methods": {"joint": method},
            }

        result = angular_xdp_envelope(
            pair_rows=[pair(0.04, 4.0), pair(0.08, 8.0)],
            methods=["joint"],
            angular_radii=[0.05, 0.10],
            deltas=[0.01],
        )
        first, second = result["rows"]
        self.assertEqual(first["evaluated_pairs"], 1)
        self.assertEqual(first["linkage_xdp_xi_point_envelope"], 4.0)
        self.assertEqual(second["evaluated_pairs"], 2)
        self.assertEqual(second["linkage_xdp_xi_point_envelope"], 8.0)


if __name__ == "__main__":
    unittest.main()
