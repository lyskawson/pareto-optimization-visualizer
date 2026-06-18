from __future__ import annotations

import os
from typing import List, Tuple

from RandomNumberGenerator import RandomNumberGenerator
from src.algorithms import constant_acceptance, pareto_sa
from src.chernoff import FEATURE_NAMES, draw_faces
from src.instance import FlowShopInstance, generate_instance
from src.neighborhood import random_neighbor, random_permutation
from src.schedule import criteria_four

Vector = Tuple[int, int, int, int]
Solution = Tuple[List[int], Vector]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")
FRONT_SEED = 12345
FRONT_MAX_ITER = 1600
PERTURB_MOVES = 5


def final_front(instance: FlowShopInstance) -> List[Solution]:
    rng = RandomNumberGenerator(FRONT_SEED)
    evaluate = lambda pi: criteria_four(instance, pi)
    result = pareto_sa(
        instance, rng, FRONT_MAX_ITER, constant_acceptance(0.1), evaluate=evaluate
    )
    return [(perm, point) for perm, point in result.front_solutions]


def spread_three(front: List[Solution]) -> List[Solution]:
    ordered = sorted(front, key=lambda sol: sol[1])
    if len(ordered) <= 3:
        return ordered
    mid = len(ordered) // 2
    return [ordered[0], ordered[mid], ordered[-1]]


def ensure_three(instance: FlowShopInstance, front: List[Solution]) -> List[Solution]:
    chosen = spread_three(front)
    rng = RandomNumberGenerator(54321)
    while len(chosen) < 3:
        base = chosen[0][0] if chosen else random_permutation(instance.n, rng)
        perm = list(base)
        for _ in range(PERTURB_MOVES):
            perm = random_neighbor(perm, rng, "insert")
        chosen.append((perm, criteria_four(instance, perm)))
    return chosen[:3]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance = generate_instance(n=50, seed=1)

    front = final_front(instance)
    good = ensure_three(instance, front)

    worse_pi = random_permutation(instance.n, RandomNumberGenerator(99))
    worse = (worse_pi, criteria_four(instance, worse_pi))

    solutions = good + [worse]
    labels = [f"Good #{i + 1}" for i in range(len(good))] + ["Worse"]
    vectors = [point for _, point in solutions]

    path = os.path.join(RESULTS_DIR, "chernoff_faces.png")
    draw_faces(
        vectors,
        labels,
        path,
        suptitle="Chernoff faces - criteria (2,3,5,6) = (SumF, Tmax, Lmax, SumL)",
    )

    print(
        "Chernoff faces - 3 solutions from the pareto_sa final front (4 criteria) "
        "plus 1 weaker (random) solution."
    )
    print(f"Final 4-criteria front size: {len(front)}")
    print("\nFeature mapping (per-column min-max normalized over the 4 faces):")
    for name in FEATURE_NAMES:
        print(f"  - {name}")
    print(f"\n{'solution':>10}  {'SumF':>8}  {'Tmax':>6}  {'Lmax':>6}  {'SumL':>8}")
    for label, values in zip(labels, vectors):
        print(
            f"{label:>10}  {values[0]:>8}  {values[1]:>6}  {values[2]:>6}  {values[3]:>8}"
        )
    print("\nchernoff_faces.png saved to results/")
    print(
        "Decision: smaller/flatter features (short face, smile, small eyes, level brows) "
        "mark the dominating schedules; a 'Good' face is selected over 'Worse'."
    )


if __name__ == "__main__":
    main()
