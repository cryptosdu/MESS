"""XDP accounting used by the MESS security experiments.

This module deliberately keeps two guarantees separate.

1. ``original_lshrr_xdp`` implements Proposition 5 of Fernandes, Kawamoto,
   and Murakami (ESORICS 2021).  It requires independent LSH bits satisfying

       Pr[h(x) != h(x')] = d(x, x').

2. ``fixed_hash_xdp`` conditions on already trained hash mappings.  Its
   distance is the observed Hamming distance, not the original angular
   distance.

The distinction prevents a per-Hamming-bit coefficient from being printed as
if it were a total angular-XDP privacy budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import binom


def check_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 0.5:
        raise ValueError("%s must lie in (0, 0.5)" % name)
    return value


def rr_epsilon_bit(flip_probability: float) -> float:
    """Return the per-differing-bit randomized-response privacy cost."""

    p = check_probability(flip_probability, "flip_probability")
    return math.log((1.0 - p) / p)


def angular_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return normalized angular distance acos(cosine)/pi in [0, 1]."""

    lhs = np.asarray(x, dtype=np.float64).reshape(-1)
    rhs = np.asarray(y, dtype=np.float64).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError("angular-distance inputs must have identical shapes")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("angular distance is undefined for a zero vector")
    cosine = float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))
    cosine = min(1.0, max(-1.0, cosine))
    return math.acos(cosine) / math.pi


def bernoulli_kl(q: float, p: float) -> float:
    """KL(Bernoulli(q) || Bernoulli(p)), including stable endpoints."""

    q = float(q)
    p = float(p)
    if not 0.0 <= q <= 1.0 or not 0.0 <= p <= 1.0:
        raise ValueError("Bernoulli parameters must lie in [0, 1]")
    if q == p:
        return 0.0
    if p == 0.0:
        return 0.0 if q == 0.0 else float("inf")
    if p == 1.0:
        return 0.0 if q == 1.0 else float("inf")
    first = 0.0 if q == 0.0 else q * math.log(q / p)
    second = 0.0 if q == 1.0 else (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    return first + second


def proposition5_alpha(
    hash_bits: int,
    input_distance: float,
    target_delta: float,
    iterations: int = 120,
) -> Tuple[float, float, bool]:
    """Invert Proposition 5's Chernoff bound.

    Returns ``(alpha, achieved_delta_bound, pure_fallback)``.  The fallback is
    needed only when the requested tail is beyond the range of the stated
    Chernoff expression.
    """

    k = int(hash_bits)
    d = float(input_distance)
    delta = float(target_delta)
    if k <= 0:
        raise ValueError("hash_bits must be positive")
    if not 0.0 <= d <= 1.0:
        raise ValueError("input_distance must lie in [0, 1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1)")
    if d == 0.0:
        return 0.0, 0.0, False
    if d == 1.0:
        return 0.0, 0.0, True

    target_kl = -math.log(delta) / k
    maximum_kl = math.log(1.0 / d)  # KL(1 || d)
    if target_kl >= maximum_kl:
        return 1.0 - d, math.exp(-k * maximum_kl), True

    lo = d
    hi = 1.0
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if bernoulli_kl(mid, d) >= target_kl:
            hi = mid
        else:
            lo = mid
    alpha = hi - d
    achieved = math.exp(-k * bernoulli_kl(hi, d))
    return alpha, achieved, False


@dataclass(frozen=True)
class OriginalXDPResult:
    hash_bits: int
    angular_distance: float
    flip_probability: float
    target_delta: float
    epsilon_bit: float
    alpha: float
    chernoff_mismatch_threshold: float
    xdp_xi_proposition5: float
    xdp_xi_exact_binomial: float
    exact_binomial_mismatch_threshold: int
    exact_binomial_tail: float
    ordinary_ldp_epsilon: float
    used_pure_fallback: bool

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["definition"] = (
            "random-LSH angular XDP; requires independent LSH bits with "
            "Pr[h(x)!=h(x')]=d_theta"
        )
        return result


def exact_binomial_threshold(
    hash_bits: int, input_distance: float, target_delta: float
) -> Tuple[int, float]:
    """Smallest integer u satisfying Pr[Binomial(k,d) > u] <= delta."""

    k = int(hash_bits)
    d = float(input_distance)
    delta = float(target_delta)
    if k <= 0:
        raise ValueError("hash_bits must be positive")
    if not 0.0 <= d <= 1.0:
        raise ValueError("input_distance must lie in [0, 1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1)")
    for threshold in range(k + 1):
        tail = float(binom.sf(threshold, k, d))
        if tail <= delta:
            return threshold, tail
    return k, 0.0


def original_lshrr_xdp(
    flip_probability: float,
    hash_bits: int,
    input_distance: float,
    target_delta: float,
) -> OriginalXDPResult:
    """Calculate the paper's Proposition-5 XDP and an exact-binomial variant."""

    p = check_probability(flip_probability, "flip_probability")
    k = int(hash_bits)
    d = float(input_distance)
    epsilon_bit = rr_epsilon_bit(p)
    alpha, _, fallback = proposition5_alpha(k, d, target_delta)
    ldp = k * epsilon_bit
    proposition5_xi = ldp if fallback else epsilon_bit * k * (d + alpha)
    exact_u, exact_tail = exact_binomial_threshold(k, d, target_delta)
    exact_xi = epsilon_bit * exact_u
    return OriginalXDPResult(
        hash_bits=k,
        angular_distance=d,
        flip_probability=p,
        target_delta=float(target_delta),
        epsilon_bit=epsilon_bit,
        alpha=alpha,
        chernoff_mismatch_threshold=k * (d + alpha),
        xdp_xi_proposition5=proposition5_xi,
        xdp_xi_exact_binomial=exact_xi,
        exact_binomial_mismatch_threshold=exact_u,
        exact_binomial_tail=exact_tail,
        ordinary_ldp_epsilon=ldp,
        used_pure_fallback=fallback,
    )


def original_multigraph_reference(
    flip_probability: float,
    bits_per_shard: int,
    shard_counts: Sequence[int],
    input_distance: float,
    target_delta: float,
) -> List[Dict[str, object]]:
    """Apply Proposition 5 jointly to t independent random-LSH shard views."""

    rows = []
    for t in shard_counts:
        t = int(t)
        if t <= 0:
            raise ValueError("shard counts must be positive")
        row = original_lshrr_xdp(
            flip_probability=flip_probability,
            hash_bits=t * int(bits_per_shard),
            input_distance=input_distance,
            target_delta=target_delta,
        ).to_dict()
        row["observed_shards"] = t
        row["bits_per_shard"] = int(bits_per_shard)
        rows.append(row)
    return rows


def paper_regression_rows() -> List[Dict[str, object]]:
    """Return deterministic checks reproducing Table 1 of the XDP paper."""

    expected = {
        0.05: {
            10: [3, 14, 28, 55],
            20: [4, 20, 40, 79],
            50: [6, 30, 60, 120],
        },
        0.10: {
            10: [2, 10, 21, 42],
            20: [3, 14, 28, 57],
            50: [4, 20, 40, 80],
        },
    }
    rows = []
    for d, by_bits in expected.items():
        for k, expected_ldp in by_bits.items():
            alpha, _, fallback = proposition5_alpha(k, d, 0.01)
            denominator = float(k) if fallback else k * (d + alpha)
            for target_xi, expected_value in zip((1.0, 5.0, 10.0, 20.0), expected_ldp):
                epsilon_bit = target_xi / denominator
                observed_ldp = int(round(k * epsilon_bit))
                rows.append(
                    {
                        "angular_distance": d,
                        "hash_bits": k,
                        "delta": 0.01,
                        "target_xi": target_xi,
                        "expected_rounded_ldp": expected_value,
                        "observed_rounded_ldp": observed_ldp,
                        "passed": observed_ldp == expected_value,
                    }
                )
    return rows

