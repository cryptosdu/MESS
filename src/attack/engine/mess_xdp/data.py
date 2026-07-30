"""Dataset, frozen IsoHash, and angular challenge-pair utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    directory: str
    base_candidates: Tuple[str, ...]
    feature_type: str
    dimension: int


DATASETS = {
    "sift": DatasetPreset(
        "sift", "dataset/sift", ("base.fvecs",), "fvecs", 128
    ),
    "laion": DatasetPreset(
        "laion",
        "dataset/laion1m",
        ("100k/laion_base.fvecs",),
        "fvecs",
        512,
    ),
    "trip": DatasetPreset(
        "trip", "dataset/trip_distilbert", ("passages.fvecs",), "fvecs", 768
    ),
    "msmarco": DatasetPreset(
        "msmarco",
        "dataset/msmarco_bert",
        ("passages.fvecs",),
        "fvecs",
        768,
    ),
    "sift100m": DatasetPreset(
        "sift100m",
        "dataset/sift100m",
        ("base.bin",),
        "auto_bin",
        128,
    ),
}


WEIGHT_CANDIDATES = ("isohash_weights.bin",)


def locate_data_root(explicit: str = "") -> Path:
    """Find the repository's Compass-compatible ``data`` directory."""

    if explicit:
        return Path(explicit).expanduser().resolve()
    environment = os.environ.get("MESS_DATA_ROOT", "").strip()
    if environment:
        return Path(environment).expanduser().resolve()

    starts = [Path.cwd().resolve(), Path(__file__).resolve()]
    visited = set()
    for start in starts:
        for ancestor in (start, *start.parents):
            key = str(ancestor)
            if key in visited:
                continue
            visited.add(key)
            if ancestor.name == "data" and (ancestor / "dataset").is_dir():
                return ancestor.resolve()
            candidate = ancestor / "data"
            if (candidate / "dataset").is_dir():
                return candidate.resolve()
    return (Path.cwd().resolve() / "data").resolve()


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_dataset(
    dataset: str,
    data_root: str = "",
    dataset_file: str = "",
    weight_file: str = "",
) -> Tuple[DatasetPreset, Path, Path, Path]:
    if dataset not in DATASETS:
        raise KeyError("unknown dataset %r" % dataset)
    preset = DATASETS[dataset]
    root = locate_data_root(data_root)
    directory = root / preset.directory

    if dataset_file:
        base = Path(dataset_file).expanduser()
        if not base.is_absolute():
            base = directory / base
        base = base.resolve()
    else:
        base = _first_existing(directory, preset.base_candidates)
    if base is None or not base.is_file():
        expected = "\n  ".join(str(directory / name) for name in preset.base_candidates)
        raise FileNotFoundError(
            "Dataset file was not found. Checked:\n  %s\n"
            "Run from the repository or pass its relative data path with "
            "--data-root." % expected
        )

    if weight_file:
        weight = Path(weight_file).expanduser()
        if not weight.is_absolute():
            weight = directory / weight
        weight = weight.resolve()
    else:
        weight = _first_existing(directory, WEIGHT_CANDIDATES)
    if weight is None or not weight.is_file():
        expected = "\n  ".join(str(directory / name) for name in WEIGHT_CANDIDATES)
        raise FileNotFoundError(
            "IsoHash model was not found. Checked:\n  %s\n"
            "Use --weight-file isohash_weights.bin when needed." % expected
        )
    return preset, directory.resolve(), base.resolve(), weight.resolve()


def discover_datasets(data_root: str = "") -> Dict[str, Dict[str, object]]:
    root = locate_data_root(data_root)
    result = {}
    for name, preset in DATASETS.items():
        directory = root / preset.directory
        base = _first_existing(directory, preset.base_candidates)
        weight = _first_existing(directory, WEIGHT_CANDIDATES)
        result[name] = {
            "directory": str(directory),
            "dataset_file": "" if base is None else str(base),
            "weight_file": "" if weight is None else str(weight),
            "ready": base is not None and weight is not None,
        }
    return result


def load_fvecs(path: Path, maximum_vectors: int) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        first = np.fromfile(handle, dtype="<i4", count=1)
    if first.size != 1 or int(first[0]) <= 0:
        raise ValueError("invalid fvecs header: %s" % path)
    dimension = int(first[0])
    stride = 4 * (dimension + 1)
    size = path.stat().st_size
    if size % stride:
        raise ValueError("file size is not divisible by the fvecs stride: %s" % path)
    count = size // stride
    take = min(count, int(maximum_vectors)) if maximum_vectors > 0 else count
    raw = np.memmap(path, dtype="<i4", mode="r", shape=(count, dimension + 1))
    if not np.all(raw[:take, 0] == dimension):
        raise ValueError("inconsistent fvecs dimensions: %s" % path)
    return raw[:take, 1:].view("<f4").astype(np.float32, copy=True)


def load_auto_bin(path: Path, default_dimension: int, maximum_vectors: int) -> np.ndarray:
    """Load common uint8/float32 ANN ``.bin`` layouts."""

    path = Path(path)
    size = path.stat().st_size
    if size < 1:
        raise ValueError("empty binary vector file: %s" % path)

    header = np.fromfile(path, dtype="<u4", count=2) if size >= 8 else np.asarray([])
    if header.size == 2:
        count = int(header[0])
        dimension = int(header[1])
        if count > 0 and dimension > 0:
            if 8 + count * dimension == size:
                take = min(count, maximum_vectors) if maximum_vectors > 0 else count
                matrix = np.memmap(
                    path, dtype=np.uint8, mode="r", offset=8, shape=(count, dimension)
                )
                return np.asarray(matrix[:take], dtype=np.float32)
            if 8 + 4 * count * dimension == size:
                take = min(count, maximum_vectors) if maximum_vectors > 0 else count
                matrix = np.memmap(
                    path, dtype="<f4", mode="r", offset=8, shape=(count, dimension)
                )
                return np.asarray(matrix[:take], dtype=np.float32)

    dimension = int(default_dimension)
    if size % dimension:
        raise ValueError(
            "cannot infer binary vector layout for %s (size=%d, dimension=%d)"
            % (path, size, dimension)
        )
    count = size // dimension
    take = min(count, maximum_vectors) if maximum_vectors > 0 else count
    matrix = np.memmap(path, dtype=np.uint8, mode="r", shape=(count, dimension))
    return np.asarray(matrix[:take], dtype=np.float32)


def load_vectors(
    preset: DatasetPreset, path: Path, maximum_vectors: int
) -> np.ndarray:
    if preset.feature_type == "fvecs":
        matrix = load_fvecs(path, maximum_vectors)
    elif preset.feature_type == "auto_bin":
        matrix = load_auto_bin(path, preset.dimension, maximum_vectors)
    else:
        raise ValueError("unsupported feature type %s" % preset.feature_type)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("at least two vectors are required")
    return matrix


@dataclass(frozen=True)
class IsoHashModel:
    mean: np.ndarray
    weights: List[np.ndarray]
    thresholds: List[np.ndarray]
    source_bits: int
    requested_total_bits: int
    tiled_repetitions: int
    has_distinct_shard_blocks: bool
    layout: str

    def metadata(self) -> Dict[str, object]:
        return {
            "dimension": int(self.mean.size),
            "num_shards": len(self.weights),
            "bits_per_shard": int(self.thresholds[0].size),
            "source_bits": self.source_bits,
            "requested_total_bits": self.requested_total_bits,
            "tiled_repetitions": self.tiled_repetitions,
            "has_distinct_shard_blocks": self.has_distinct_shard_blocks,
            "layout": self.layout,
        }


def load_isohash(
    path: Path, dimension: int, num_shards: int, bits_per_shard: int
) -> IsoHashModel:
    """Parse mean + row-major bit projections + thresholds.

    A model with fewer bits may be tiled for compatibility.  This is reported
    explicitly because tiled shard blocks are not independent mappings.
    """

    array = np.fromfile(path, dtype="<f4")
    dimension = int(dimension)
    requested = int(num_shards) * int(bits_per_shard)
    candidates = []

    if array.size > dimension and (array.size - dimension) % (dimension + 1) == 0:
        bits = (array.size - dimension) // (dimension + 1)
        candidates.append((True, int(bits)))
    if array.size % (dimension + 1) == 0:
        bits = array.size // (dimension + 1)
        candidates.append((False, int(bits)))

    compatible = [
        (has_mean, bits)
        for has_mean, bits in candidates
        if bits > 0 and requested % bits == 0
    ]
    if not compatible:
        raise ValueError(
            "Cannot parse IsoHash model %s: %d float32 values, dimension=%d, "
            "requested_bits=%d" % (path, array.size, dimension, requested)
        )
    compatible.sort(key=lambda item: (item[1] == requested, item[1], item[0]), reverse=True)
    has_mean, source_bits = compatible[0]
    offset = dimension if has_mean else 0
    mean = (
        array[:dimension].astype(np.float32, copy=True)
        if has_mean
        else np.zeros(dimension, dtype=np.float32)
    )
    weight_end = offset + source_bits * dimension
    saved_weights = array[offset:weight_end].reshape(source_bits, dimension)
    source_weights = saved_weights.T.astype(np.float32, copy=True)
    source_thresholds = array[weight_end : weight_end + source_bits].astype(
        np.float32, copy=True
    )
    repetitions = requested // source_bits
    full_weights = np.tile(source_weights, (1, repetitions))
    full_thresholds = np.tile(source_thresholds, repetitions)
    weights = []
    thresholds = []
    for shard in range(int(num_shards)):
        start = shard * int(bits_per_shard)
        end = start + int(bits_per_shard)
        weights.append(full_weights[:, start:end].copy())
        thresholds.append(full_thresholds[start:end].copy())
    layout = (
        ("mean + " if has_mean else "")
        + "W[%d,%d] + thresholds[%d]" % (source_bits, dimension, source_bits)
    )
    return IsoHashModel(
        mean=mean,
        weights=weights,
        thresholds=thresholds,
        source_bits=source_bits,
        requested_total_bits=requested,
        tiled_repetitions=repetitions,
        has_distinct_shard_blocks=source_bits >= requested,
        layout=layout,
    )


def hash_batch(
    vectors: np.ndarray,
    mean: np.ndarray,
    weight: np.ndarray,
    threshold: np.ndarray,
    normalize_centered: bool = False,
) -> np.ndarray:
    centered = np.asarray(vectors, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    if normalize_centered:
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        centered = centered / np.maximum(norms, 1e-12)
    return ((centered @ weight) >= threshold).astype(np.uint8)


def pair_shard_distances(
    x0: np.ndarray,
    x1: np.ndarray,
    model: IsoHashModel,
    normalize_centered: bool = False,
) -> List[int]:
    pair = np.vstack([x0, x1]).astype(np.float32)
    distances = []
    for weight, threshold in zip(model.weights, model.thresholds):
        codes = hash_batch(pair, model.mean, weight, threshold, normalize_centered)
        distances.append(int(np.count_nonzero(codes[0] != codes[1])))
    return distances


def normalized_rows(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    normalized = np.zeros_like(vectors, dtype=np.float32)
    normalized[valid] = vectors[valid] / norms[valid, None]
    return normalized, valid


@dataclass(frozen=True)
class ChallengePair:
    first_id: int
    second_id: int
    group: str
    target_angular_distance: Optional[float]
    angular_distance: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def select_angular_pairs(
    vectors: np.ndarray,
    targets: Sequence[float],
    pairs_per_target: int,
    pool_size: int,
    anchor_count: int,
    seed: int,
    include_nearest: bool = True,
) -> List[ChallengePair]:
    """Select real pairs by normalized angular distance.

    Selection is deterministic for a fixed seed.  For every target distance,
    the closest available real pairs are selected and their actual distances
    are always reported.  This avoids silently relabeling an L2-nearest pair as
    an angular-XDP pair.
    """

    if vectors.shape[0] < 2:
        raise ValueError("at least two vectors are required")
    if int(pairs_per_target) < 1:
        raise ValueError("pairs_per_target must be positive")
    rng = np.random.default_rng(int(seed))
    size = min(int(pool_size), vectors.shape[0])
    ids = rng.choice(vectors.shape[0], size=size, replace=False)
    pool, valid = normalized_rows(vectors[ids])
    if int(valid.sum()) < 2:
        raise ValueError("not enough nonzero vectors for angular pairs")
    ids = ids[valid]
    pool = pool[valid]
    size = pool.shape[0]
    anchors = min(int(anchor_count), size)
    anchor_positions = rng.choice(size, size=anchors, replace=False)

    candidates = {"nearest": []}
    for target in targets:
        target = float(target)
        if not 0.0 <= target <= 1.0:
            raise ValueError("target angular distances must lie in [0, 1]")
        candidates["target_%.6g" % target] = []

    block_size = 32
    retain = max(4, int(pairs_per_target))
    for start in range(0, anchors, block_size):
        positions = anchor_positions[start : start + block_size]
        cosine = np.clip(pool[positions] @ pool.T, -1.0, 1.0)
        distances = np.arccos(cosine) / math.pi
        for row, anchor_position in enumerate(positions.tolist()):
            distances[row, anchor_position] = np.inf
            nearest_position = int(np.argmin(distances[row]))
            candidates["nearest"].append(
                (
                    float(distances[row, nearest_position]),
                    int(ids[anchor_position]),
                    int(ids[nearest_position]),
                )
            )
            for target in targets:
                key = "target_%.6g" % float(target)
                errors = np.abs(distances[row] - float(target))
                take = min(retain, size - 1)
                positions_for_target = np.argpartition(errors, take - 1)[:take]
                for position in positions_for_target.tolist():
                    if position == anchor_position or not math.isfinite(
                        float(distances[row, position])
                    ):
                        continue
                    candidates[key].append(
                        (
                            abs(float(distances[row, position]) - float(target)),
                            float(distances[row, position]),
                            int(ids[anchor_position]),
                            int(ids[position]),
                        )
                    )

    output = []
    used = set()

    def add_pair(first: int, second: int, group: str, target: Optional[float], distance: float) -> None:
        key = tuple(sorted((int(first), int(second))))
        if key in used:
            return
        used.add(key)
        output.append(
            ChallengePair(
                first_id=key[0],
                second_id=key[1],
                group=group,
                target_angular_distance=target,
                angular_distance=float(distance),
            )
        )

    if include_nearest:
        nearest = sorted(
            candidates["nearest"],
            key=lambda item: (item[0], min(item[1], item[2]), max(item[1], item[2])),
        )
        added = 0
        for distance, first, second in nearest:
            before = len(output)
            add_pair(first, second, "semantic_nearest", None, distance)
            added += len(output) - before
            if added >= int(pairs_per_target):
                break

    for target in targets:
        key = "target_%.6g" % float(target)
        rows = sorted(
            candidates[key],
            key=lambda item: (
                item[0],
                item[1],
                min(item[2], item[3]),
                max(item[2], item[3]),
            ),
        )
        added = 0
        for _, distance, first, second in rows:
            before = len(output)
            add_pair(first, second, key, float(target), distance)
            added += len(output) - before
            if added >= int(pairs_per_target):
                break
        if added < int(pairs_per_target):
            raise RuntimeError(
                "only %d unique pairs were available for target %.6g; "
                "increase --pair-pool-size or --pair-anchors" % (added, target)
            )
    return output


def file_identity(path: Path, full_hash_limit: int = 64 * 1024 * 1024) -> Dict[str, object]:
    """Return a reproducibility fingerprint without scanning huge datasets.

    Files up to 64 MiB receive a full SHA-256.  Larger files use their size and
    SHA-256 over the first and last MiB.  The mode is recorded explicitly.
    """

    path = Path(path)
    size = int(path.stat().st_size)
    digest = hashlib.sha256()
    if size <= int(full_hash_limit):
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        mode = "full_sha256"
    else:
        with path.open("rb") as handle:
            first = handle.read(1024 * 1024)
            handle.seek(max(0, size - 1024 * 1024))
            last = handle.read(1024 * 1024)
        digest.update(str(size).encode("ascii"))
        digest.update(first)
        digest.update(last)
        mode = "size_plus_first_last_1MiB_sha256"
    return {"size_bytes": size, "fingerprint_mode": mode, "sha256": digest.hexdigest()}
