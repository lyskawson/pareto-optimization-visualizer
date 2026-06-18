from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from RandomNumberGenerator import RandomNumberGenerator
from src.instance import FlowShopInstance
from src.neighborhood import random_neighbor, random_permutation
from src.pareto import Point, dominates, pareto_front
from src.schedule import criteria_three, criteria_two

Evaluate = Callable[[List[int]], Point]


def constant_acceptance(probability: float = 0.1) -> Callable[[int], float]:
    return lambda it: probability


def geometric_acceptance(base: float = 0.995) -> Callable[[int], float]:
    return lambda it: base ** it


ACCEPTANCE = {"constant": constant_acceptance, "geometric": geometric_acceptance}


@dataclass
class ParetoResult:
    population: List[Point]
    front: List[Point]
    front_solutions: List[Tuple[List[int], Point]]


def pareto_sa(
    instance: FlowShopInstance,
    rng: RandomNumberGenerator,
    max_iter: int,
    p_accept: Callable[[int], float],
    move: str = "insert",
    evaluate: Optional[Evaluate] = None,
) -> ParetoResult:
    if evaluate is None:
        evaluate = lambda pi: criteria_two(instance, pi)
    x = random_permutation(instance.n, rng)
    fx = evaluate(x)
    archive: List[Tuple[List[int], Point]] = [(list(x), fx)]
    for it in range(max_iter):
        x2 = random_neighbor(x, rng, move)
        fx2 = evaluate(x2)
        if dominates(fx2, fx):
            x, fx = x2, fx2
            archive.append((list(x2), fx2))
        elif rng.nextFloat(0.0, 1.0) < p_accept(it):
            x, fx = x2, fx2
            archive.append((list(x2), fx2))
    population = [point for _, point in archive]
    front = pareto_front(population)
    front_set = set(front)
    seen: set = set()
    front_solutions: List[Tuple[List[int], Point]] = []
    for perm, point in archive:
        if point in front_set and point not in seen:
            seen.add(point)
            front_solutions.append((perm, point))
    return ParetoResult(
        population=population, front=front, front_solutions=front_solutions
    )


def scalarize(
    values: Sequence[float], coefficients: Sequence[float]
) -> float:
    return sum(c * v for c, v in zip(coefficients, values))


@dataclass
class ScalarResult:
    permutation: List[int]
    value: float


def scalarized_sa(
    instance: FlowShopInstance,
    rng: RandomNumberGenerator,
    max_iter: int,
    coefficients: Sequence[float],
    p_accept: Callable[[int], float],
    move: str = "insert",
    on_step: Optional[Callable[[int, List[int]], None]] = None,
) -> ScalarResult:
    score = lambda pi: scalarize(criteria_three(instance, pi), coefficients)
    x = random_permutation(instance.n, rng)
    sx = score(x)
    best_x, best_s = list(x), sx
    for it in range(max_iter):
        x2 = random_neighbor(x, rng, move)
        sx2 = score(x2)
        if sx2 < sx:
            x, sx = x2, sx2
        elif rng.nextFloat(0.0, 1.0) < p_accept(it):
            x, sx = x2, sx2
        if sx < best_s:
            best_x, best_s = list(x), sx
        if on_step is not None:
            on_step(it + 1, best_x)
    return ScalarResult(permutation=best_x, value=best_s)
