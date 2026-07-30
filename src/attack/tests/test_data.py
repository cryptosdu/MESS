import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from mess_xdp.data import (  # noqa: E402
    load_isohash,
    pair_shard_distances,
    select_angular_pairs,
)


class DataTests(unittest.TestCase):
    def test_isohash_mean_layout(self):
        dimension = 3
        shards = 2
        bits = 2
        mean = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        weights = np.arange(shards * bits * dimension, dtype=np.float32).reshape(
            shards * bits, dimension
        )
        thresholds = np.zeros(shards * bits, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            np.concatenate([mean, weights.reshape(-1), thresholds]).tofile(path)
            model = load_isohash(path, dimension, shards, bits)
        self.assertEqual(model.source_bits, 4)
        self.assertTrue(model.has_distinct_shard_blocks)
        self.assertEqual(len(model.weights), 2)

    def test_pair_distance(self):
        dimension = 2
        mean = np.zeros(dimension, dtype=np.float32)
        saved = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        threshold = np.zeros(2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            np.concatenate([mean, saved.reshape(-1), threshold]).tofile(path)
            model = load_isohash(path, dimension, 1, 2)
            distances = pair_shard_distances(
                np.asarray([1.0, 1.0]),
                np.asarray([-1.0, -1.0]),
                model,
            )
        self.assertEqual(distances, [2])

    def test_angular_pair_selection_reports_actual_distance(self):
        angles = np.linspace(0.0, 0.5, 40)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        pairs = select_angular_pairs(
            vectors,
            targets=[0.05],
            pairs_per_target=1,
            pool_size=40,
            anchor_count=20,
            seed=7,
        )
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(0.0 <= pair.angular_distance <= 1.0 for pair in pairs))


if __name__ == "__main__":
    unittest.main()

