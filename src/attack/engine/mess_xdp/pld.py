"""Fixed-mapping route accounting and exact randomized-response PLDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from .xdp import rr_epsilon_bit


def uniform_route_sum_distribution(
    shard_distances: Sequence[int], selected_shards: int
) -> Dict[int, float]:
    """Distribution of the sum for a uniform t-of-M shard route.

    Dynamic programming avoids enumerating all ``binom(M,t)`` routes.
    """

    values = [int(value) for value in shard_distances]
    if any(value < 0 for value in values):
        raise ValueError("shard distances cannot be negative")
    m = len(values)
    t = int(selected_shards)
    if not 0 <= t <= m:
        raise ValueError("selected_shards must lie between 0 and len(distances)")

    states: List[Dict[int, int]] = [dict() for _ in range(t + 1)]
    states[0][0] = 1
    for index, distance in enumerate(values, start=1):
        upper = min(index, t)
        for chosen in range(upper, 0, -1):
            previous = states[chosen - 1]
            current = states[chosen]
            for total, count in previous.items():
                updated = total + distance
                current[updated] = current.get(updated, 0) + count

    denominator = math.comb(m, t)
    if denominator <= 0:
        raise RuntimeError("invalid route count")
    distribution = {
        total: count / float(denominator) for total, count in states[t].items()
    }
    normalizer = sum(distribution.values())
    return {total: probability / normalizer for total, probability in distribution.items()}


def route_mismatch_threshold(
    route_distribution: Mapping[int, float], target_delta: float
) -> Tuple[int, float]:
    """Smallest u satisfying Pr[U > u] <= target_delta."""

    delta = float(target_delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1)")
    if not route_distribution:
        raise ValueError("route_distribution cannot be empty")
    items = sorted((int(u), float(probability)) for u, probability in route_distribution.items())
    total = sum(probability for _, probability in items)
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("route probabilities must sum to one")
    remaining = total
    for u, probability in items:
        remaining -= probability
        if remaining <= delta + 1e-15:
            return u, max(0.0, remaining)
    return items[-1][0], 0.0


def _binomial_mass(n: int, success_probability: float) -> np.ndarray:
    if n == 0:
        return np.asarray([1.0], dtype=np.float64)
    z = np.arange(n + 1, dtype=np.float64)
    log_choose = gammaln(n + 1.0) - gammaln(z + 1.0) - gammaln(n - z + 1.0)
    log_mass = (
        log_choose
        + z * math.log(success_probability)
        + (n - z) * math.log(1.0 - success_probability)
    )
    mass = np.exp(log_mass)
    mass /= max(float(mass.sum()), np.finfo(np.float64).tiny)
    return mass


@dataclass(frozen=True)
class DiscretePLD:
    """Privacy-loss distribution under the numerator world."""

    losses: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        losses = np.asarray(self.losses, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if losses.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("losses and probabilities must be one-dimensional")
        if losses.size != probabilities.size:
            raise ValueError("losses and probabilities must have the same length")
        if np.any(probabilities < -1e-15):
            raise ValueError("PLD contains negative probability")
        if not math.isclose(
            float(probabilities.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("PLD probabilities must sum to one")

    @property
    def pure_epsilon(self) -> float:
        return max(0.0, float(np.max(self.losses, initial=0.0)))

    def delta(self, epsilon: float) -> float:
        """Exact hockey-stick profile E[(1-exp(epsilon-L))_+]."""

        epsilon = float(epsilon)
        mask = self.losses > epsilon
        if not np.any(mask):
            return 0.0
        factors = -np.expm1(np.clip(epsilon - self.losses[mask], -745.0, 0.0))
        return float(np.sum(self.probabilities[mask] * factors))

    def tail_probability(self, epsilon: float) -> float:
        return float(np.sum(self.probabilities[self.losses > float(epsilon)]))

    def epsilon_for_delta(self, target_delta: float, iterations: int = 100) -> float:
        delta = float(target_delta)
        if not 0.0 <= delta < 1.0:
            raise ValueError("target_delta must lie in [0, 1)")
        if self.delta(0.0) <= delta:
            return 0.0
        lo = 0.0
        hi = self.pure_epsilon
        for _ in range(int(iterations)):
            mid = 0.5 * (lo + hi)
            if self.delta(mid) <= delta:
                hi = mid
            else:
                lo = mid
        return hi

    def profile(self, epsilons: Iterable[float]) -> List[Dict[str, float]]:
        return [
            {
                "epsilon": float(epsilon),
                "delta": self.delta(float(epsilon)),
                "labelled_release_tail_probability": self.tail_probability(float(epsilon)),
            }
            for epsilon in epsilons
        ]


def route_mixture_rr_pld(
    route_distribution: Mapping[int, float], flip_probability: float
) -> DiscretePLD:
    """Exact PLD for a route-labelled mixture of RR releases.

    For ``u`` differing clean bits, ``L=(2Z-u)*eta`` under world 0 with
    ``Z ~ Binomial(u,1-p)``.  Since routing is independent of the protected
    value and is visible to the server, the route-labelled PLD is the weighted
    mixture over ``u``.
    """

    if not route_distribution:
        raise ValueError("route_distribution cannot be empty")
    eta = rr_epsilon_bit(flip_probability)
    maximum = max(int(value) for value in route_distribution)
    mass = np.zeros(2 * maximum + 1, dtype=np.float64)
    offset = maximum
    for u, route_probability in route_distribution.items():
        u = int(u)
        route_probability = float(route_probability)
        z_mass = _binomial_mass(u, 1.0 - float(flip_probability))
        multipliers = 2 * np.arange(u + 1, dtype=np.int64) - u
        mass[multipliers + offset] += route_probability * z_mass
    positive = mass > 1e-18
    probabilities = mass[positive]
    probabilities /= float(probabilities.sum())
    multipliers = np.arange(-maximum, maximum + 1, dtype=np.int64)[positive]
    return DiscretePLD(multipliers.astype(np.float64) * eta, probabilities)


@dataclass(frozen=True)
class PairRouteAccounting:
    shard_distances: List[int]
    selected_shards: int
    flip_probability: float
    epsilon_bit: float
    maximum_route_distance: int
    expected_route_distance: float
    route_pxdp: List[Dict[str, float]]
    pld_approximate_dp: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["fixed_mapping_pure_xdp_epsilon"] = (
            self.epsilon_bit * self.maximum_route_distance
        )
        result["scope"] = (
            "pair-specific fixed-mapping accounting over the uniform t-of-M route"
        )
        return result


def account_fixed_mapping_pair(
    shard_distances: Sequence[int],
    selected_shards: int,
    flip_probability: float,
    deltas: Sequence[float],
) -> PairRouteAccounting:
    """Calculate route-PXDP and exact PLD budgets for one evaluated pair."""

    distribution = uniform_route_sum_distribution(shard_distances, selected_shards)
    eta = rr_epsilon_bit(flip_probability)
    pld = route_mixture_rr_pld(distribution, flip_probability)
    route_rows = []
    pld_rows = []
    for delta in deltas:
        threshold, tail = route_mismatch_threshold(distribution, float(delta))
        route_rows.append(
            {
                "delta": float(delta),
                "mismatch_threshold": float(threshold),
                "actual_route_tail": float(tail),
                "xdp_xi": float(eta * threshold),
            }
        )
        pld_rows.append(
            {
                "delta": float(delta),
                "epsilon": float(pld.epsilon_for_delta(float(delta))),
            }
        )
    expected = sum(u * probability for u, probability in distribution.items())
    maximum = max(distribution)
    return PairRouteAccounting(
        shard_distances=[int(value) for value in shard_distances],
        selected_shards=int(selected_shards),
        flip_probability=float(flip_probability),
        epsilon_bit=eta,
        maximum_route_distance=int(maximum),
        expected_route_distance=float(expected),
        route_pxdp=route_rows,
        pld_approximate_dp=pld_rows,
    )

