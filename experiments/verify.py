from __future__ import annotations

from src.algorithms import scalarize
from src.instance import FlowShopInstance, generate_instance
from src.pareto import dominates, hypervolume_2d, pareto_front
from src.schedule import completion_matrix, criteria_three, last_machine_completion


def check_small_instance() -> None:
    instance = generate_instance(n=5, seed=42)
    print("Instance n=5, seed=42")
    for i, row in enumerate(instance.p):
        print(f"  p[machine {i}] = {row}")
    print(f"  d = {instance.d}")

    pi = list(range(5))
    c = completion_matrix(instance, pi)
    print(f"  permutation = {pi}")
    print(f"  C[0] = {c[0]}")
    print(f"  C[1] = {c[1]}")
    print(f"  C[2] = {c[2]}")

    j0 = pi[0]
    assert c[0][j0] == instance.p[0][j0]
    assert c[1][j0] == instance.p[0][j0] + instance.p[1][j0]
    assert c[2][j0] == c[1][j0] + instance.p[2][j0]
    j1 = pi[1]
    assert c[0][j1] == c[0][j0] + instance.p[0][j1]
    assert c[1][j1] == max(c[0][j1], c[1][j0]) + instance.p[1][j1]
    assert c[2][j1] == max(c[1][j1], c[2][j0]) + instance.p[2][j1]
    print("  recurrence checks passed")


def check_due_date_indexing() -> None:
    ones = [1, 1, 1]
    instance = FlowShopInstance(n=3, seed=0, p=[ones, ones, ones], d=[10, 1, 5])
    pi = [2, 0, 1]
    cm = last_machine_completion(instance, pi)
    assert (cm[0], cm[1], cm[2]) == (4, 5, 3), cm

    sumf, tmax, lmax = criteria_three(instance, pi)
    assert (sumf, tmax, lmax) == (12, 4, 4), (sumf, tmax, lmax)

    position_lmax = max(cm[pi[k]] - instance.d[k] for k in range(instance.n))
    assert position_lmax == 3
    assert position_lmax != lmax
    print(
        "Due-date indexing keyed by job index, not schedule position, passed:",
        (sumf, tmax, lmax),
        "(position-keyed Lmax would be", position_lmax, ")",
    )


def check_dominance() -> None:
    assert dominates((1, 2), (2, 3))
    assert dominates((1, 2), (1, 3))
    assert not dominates((1, 2), (1, 2))
    assert not dominates((1, 3), (2, 2))
    points = [(1, 4), (2, 2), (4, 1), (3, 3), (2, 3)]
    front = sorted(pareto_front(points))
    assert front == [(1, 4), (2, 2), (4, 1)], front
    print("Dominance and front extraction passed:", front)


def check_hvi() -> None:
    area = hypervolume_2d([(2.0, 3.0)], (5.0, 6.0))
    assert area == (5.0 - 2.0) * (6.0 - 3.0), area
    two = hypervolume_2d([(1.0, 4.0), (3.0, 2.0)], (5.0, 5.0))
    assert two == (5 - 3) * (5 - 2) + (3 - 1) * (5 - 4), two
    print("2D HVI checks passed:", area, two)


def check_reproducibility() -> None:
    a = generate_instance(n=20, seed=5)
    b = generate_instance(n=20, seed=5)
    assert a == b
    assert scalarize((10, 2, 1), (0.1, 1.0, 1.0)) == 4.0
    print("Reproducibility check passed")


def main() -> None:
    check_small_instance()
    check_due_date_indexing()
    check_dominance()
    check_hvi()
    check_reproducibility()
    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
