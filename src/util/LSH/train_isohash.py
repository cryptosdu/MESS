#!/usr/bin/env python3
"""Train IsoHash models for the datasets used by MESS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    relative_directory: str
    base_file: str
    file_type: str
    dimension: int


DATASETS: Dict[str, DatasetConfig] = {
    "sift": DatasetConfig(
        "sift", "dataset/sift", "base.fvecs", "fvecs", 128
    ),
    "laion": DatasetConfig(
        "laion",
        "dataset/laion1m",
        "100k/laion_base.fvecs",
        "fvecs",
        512,
    ),
    "trip": DatasetConfig(
        "trip", "dataset/trip_distilbert", "passages.fvecs", "fvecs", 768
    ),
    "msmarco": DatasetConfig(
        "msmarco", "dataset/msmarco_bert", "passages.fvecs", "fvecs", 768
    ),
    "sift100m": DatasetConfig(
        "sift100m", "dataset/sift100m", "base.bin", "bin_uint8", 128
    ),
}


def find_data_root(explicit: str = "") -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif os.environ.get("MESS_DATA_ROOT", "").strip():
        root = Path(os.environ["MESS_DATA_ROOT"]).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[3] / "data"
    if not (root / "dataset").is_dir():
        raise FileNotFoundError(
            "Data root does not contain dataset/: %s" % root
        )
    return root


def _sample_indices(
    count: int, maximum_samples: int, rng: np.random.Generator
) -> np.ndarray:
    take = min(int(count), int(maximum_samples))
    if take <= 0:
        raise ValueError("maximum_samples must be positive")
    if take == count:
        return np.arange(count, dtype=np.int64)

    # Floyd sampling uses O(take) memory even for SIFT100M.
    selected = set()
    for upper in range(count - take, count):
        candidate = int(rng.integers(0, upper + 1))
        selected.add(upper if candidate in selected else candidate)
    return np.asarray(sorted(selected), dtype=np.int64)


def load_fvecs_sample(
    path: Path, maximum_samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    path = Path(path)
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=1)
    if header.size != 1 or int(header[0]) <= 0:
        raise ValueError("Invalid fvecs header: %s" % path)

    dimension = int(header[0])
    row_bytes = (dimension + 1) * np.dtype("<i4").itemsize
    file_size = path.stat().st_size
    if file_size % row_bytes:
        raise ValueError("Invalid fvecs file size: %s" % path)
    count = file_size // row_bytes
    if count < 2:
        raise ValueError("At least two vectors are required: %s" % path)

    indices = _sample_indices(count, maximum_samples, rng)
    raw = np.memmap(
        path, dtype="<i4", mode="r", shape=(count, dimension + 1)
    )
    sampled = np.asarray(raw[indices], dtype="<i4")
    if not np.all(sampled[:, 0] == dimension):
        raise ValueError("Inconsistent fvecs dimensions: %s" % path)
    vectors = np.array(
        sampled[:, 1:].view("<f4"), dtype=np.float32, copy=True
    )
    return vectors, count


def load_bin_uint8_sample(
    path: Path, maximum_samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    path = Path(path)
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
    if header.size != 2:
        raise ValueError("Invalid binary-vector header: %s" % path)

    count, dimension = (int(header[0]), int(header[1]))
    if count < 2 or dimension <= 0:
        raise ValueError("Invalid binary-vector dimensions: %s" % path)
    expected_size = 8 + count * dimension
    if path.stat().st_size != expected_size:
        raise ValueError(
            "Unexpected uint8 binary-vector file size: %s" % path
        )

    indices = _sample_indices(count, maximum_samples, rng)
    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=8,
        shape=(count, dimension),
    )
    return np.asarray(raw[indices], dtype=np.float32), count


def load_training_sample(
    config: DatasetConfig,
    data_root: Path,
    maximum_samples: int,
    seed: int,
) -> Tuple[np.ndarray, int, Path]:
    path = data_root / config.relative_directory / config.base_file
    if not path.is_file():
        raise FileNotFoundError("Dataset file was not found: %s" % path)

    rng = np.random.default_rng(seed)
    if config.file_type == "fvecs":
        vectors, total = load_fvecs_sample(path, maximum_samples, rng)
    elif config.file_type == "bin_uint8":
        vectors, total = load_bin_uint8_sample(path, maximum_samples, rng)
    else:
        raise ValueError("Unsupported file type: %s" % config.file_type)

    if vectors.shape[1] != config.dimension:
        raise ValueError(
            "%s has dimension %d; expected %d"
            % (path, vectors.shape[1], config.dimension)
        )
    return vectors, total, path


def train_randomized_isohash(
    vectors: np.ndarray,
    num_bits: int,
    seed: int = 42,
    threshold_batch_bits: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if threshold_batch_bits <= 0:
        raise ValueError("threshold_batch_bits must be positive")

    training = np.asarray(vectors, dtype=np.float32)
    if training.ndim != 2 or training.shape[0] < 2:
        raise ValueError("Training data must contain at least two vectors")
    dimension = training.shape[1]

    mean = np.mean(training, axis=0, dtype=np.float32)
    centered = training - mean
    covariance = (centered.T @ centered) / np.float32(
        centered.shape[0] - 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    projection = np.empty((num_bits, dimension), dtype=np.float32)
    for block, start in enumerate(range(0, num_bits, dimension)):
        end = min(start + dimension, num_bits)
        width = end - start
        rng = np.random.default_rng(seed + 100 + block)
        random_matrix = rng.standard_normal(
            (dimension, width)
        ).astype(np.float32)
        rotation, _ = np.linalg.qr(random_matrix, mode="reduced")
        block_projection = (eigenvectors @ rotation).astype(np.float32)
        projection[start:end] = block_projection.T

    thresholds = np.empty(num_bits, dtype=np.float32)
    for start in range(0, num_bits, threshold_batch_bits):
        end = min(start + threshold_batch_bits, num_bits)
        projected = centered @ projection[start:end].T
        thresholds[start:end] = np.median(projected, axis=0).astype(
            np.float32
        )
        print("    thresholds: %d/%d bits" % (end, num_bits))

    return mean, projection, thresholds


def save_model(
    output: Path,
    mean: np.ndarray,
    projection: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.asarray(mean, dtype="<f4", order="C").tofile(handle)
        np.asarray(projection, dtype="<f4", order="C").tofile(handle)
        np.asarray(thresholds, dtype="<f4", order="C").tofile(handle)
    temporary.replace(output)

    expected_values = mean.size + projection.size + thresholds.size
    if output.stat().st_size != expected_values * 4:
        raise RuntimeError("IsoHash model size validation failed: %s" % output)


def _expand_datasets(names: Iterable[str]) -> Sequence[str]:
    selected = list(names)
    if "all" in selected:
        return tuple(DATASETS)
    return tuple(dict.fromkeys(selected))


def initialize_dataset(
    config: DatasetConfig,
    data_root: Path,
    hash_bits: int,
    maximum_samples: int,
    threshold_batch_bits: int,
    seed: int,
    force: bool,
) -> Path:
    directory = data_root / config.relative_directory
    output = directory / "isohash_weights.bin"
    if output.is_file() and not force:
        expected_size = (
            config.dimension
            + hash_bits * config.dimension
            + hash_bits
        ) * 4
        if output.stat().st_size == expected_size:
            print("[%s] exists, skipping: %s" % (config.name, output))
            return output
        raise RuntimeError(
            "%s has an unexpected size; use --force to regenerate it"
            % output
        )

    vectors, total, source = load_training_sample(
        config, data_root, maximum_samples, seed
    )
    print(
        "[%s] source=%s, total=%d, training_sample=%d, dimension=%d"
        % (config.name, source, total, vectors.shape[0], vectors.shape[1])
    )
    mean, projection, thresholds = train_randomized_isohash(
        vectors,
        num_bits=hash_bits,
        seed=seed,
        threshold_batch_bits=threshold_batch_bits,
    )
    save_model(output, mean, projection, thresholds)
    print("[%s] wrote %s" % (config.name, output))
    return output


def build_parser(default_datasets: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MESS IsoHash weights."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS) + ("all",),
        default=list(default_datasets),
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument("--hash-bits", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--threshold-batch-bits", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    default_datasets: Sequence[str] = ("sift100m",),
) -> int:
    args = build_parser(default_datasets).parse_args(argv)
    if args.hash_bits <= 0 or args.max_samples < 2:
        raise ValueError("hash-bits must be positive and max-samples >= 2")

    data_root = find_data_root(args.data_root)
    selected = _expand_datasets(args.datasets)
    failures = []
    for name in selected:
        config = DATASETS[name]
        try:
            initialize_dataset(
                config=config,
                data_root=data_root,
                hash_bits=args.hash_bits,
                maximum_samples=args.max_samples,
                threshold_batch_bits=args.threshold_batch_bits,
                seed=args.seed,
                force=args.force,
            )
        except FileNotFoundError as error:
            if not args.skip_missing:
                raise
            failures.append(str(error))
            print("[%s] skipped: %s" % (name, error))

    if failures:
        print("Completed with %d missing dataset(s)." % len(failures))
    else:
        print("IsoHash initialization completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
