from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from RandomNumberGenerator import RandomNumberGenerator
from src.algorithms import constant_acceptance, scalarized_sa
from src.instance import FlowShopInstance, generate_instance
from src.neighborhood import random_permutation
from src.schedule import criteria_three

MAX_ITERS: List[int] = [100, 200, 400, 800, 1600]
RUNS = 10
SCALE_SAMPLES = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")


def estimate_coefficients(instance: FlowShopInstance) -> Tuple[float, float, float]:
    rng = RandomNumberGenerator(7)
    sums = [0.0, 0.0, 0.0]
    for _ in range(SCALE_SAMPLES):
        pi = random_permutation(instance.n, rng)
        values = criteria_three(instance, pi)
        for i in range(3):
            sums[i] += abs(values[i])
    scales = [max(1.0, s / SCALE_SAMPLES) for s in sums]
    return 1.0 / scales[0], 1.0 / scales[1], 1.0 / scales[2]


@dataclass
class IterRow:
    mean_s: float
    best_s: float
    best_raw: Tuple[int, int, int]


def run_table(
    instance: FlowShopInstance, coefficients: Tuple[float, float, float]
) -> Dict[int, IterRow]:
    accept = constant_acceptance(0.1)
    table: Dict[int, IterRow] = {}
    for idx, max_iter in enumerate(MAX_ITERS):
        total = 0.0
        best_s = float("inf")
        best_raw: Tuple[int, int, int] = (0, 0, 0)
        for run in range(RUNS):
            rng = RandomNumberGenerator(2000 + run * 100 + idx)
            result = scalarized_sa(instance, rng, max_iter, coefficients, accept)
            total += result.value
            if result.value < best_s:
                best_s = result.value
                best_raw = criteria_three(instance, result.permutation)
        table[max_iter] = IterRow(total / RUNS, best_s, best_raw)
    return table


def save_table(
    table: Dict[int, IterRow], coefficients: Tuple[float, float, float]
) -> None:
    path = os.path.join(RESULTS_DIR, "task2_scalar.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["c1", "c2", "c3"])
        writer.writerow([f"{c:.6e}" for c in coefficients])
        writer.writerow(
            ["maxIter", "mean_s_xbest", "best_s_xbest", "SumF", "Tmax", "Lmax"]
        )
        for max_iter in MAX_ITERS:
            row = table[max_iter]
            writer.writerow(
                [
                    max_iter,
                    f"{row.mean_s:.6f}",
                    f"{row.best_s:.6f}",
                    row.best_raw[0],
                    row.best_raw[1],
                    row.best_raw[2],
                ]
            )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance = generate_instance(n=50, seed=1)
    coefficients = estimate_coefficients(instance)
    table = run_table(instance, coefficients)
    save_table(table, coefficients)
    print("Task 2 - scalarized SA over", RUNS, "runs (constant p=0.1, insert move)")
    print(
        "Coefficients (1 / estimated criterion scale): "
        f"c1={coefficients[0]:.6e}, c2={coefficients[1]:.6e}, c3={coefficients[2]:.6e}"
    )
    header = f"{'maxIter':>8}  {'mean s':>12}  {'best s':>12}  {'best (SumF, Tmax, Lmax)':>26}"
    print(header)
    for max_iter in MAX_ITERS:
        row = table[max_iter]
        raw = f"({row.best_raw[0]}, {row.best_raw[1]}, {row.best_raw[2]})"
        print(f"{max_iter:>8}  {row.mean_s:>12.6f}  {row.best_s:>12.6f}  {raw:>26}")
    print("\ntask2_scalar.csv saved to results/")


if __name__ == "__main__":
    main()
