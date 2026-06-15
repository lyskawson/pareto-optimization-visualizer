from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RandomNumberGenerator import RandomNumberGenerator
from src.algorithms import ParetoResult, constant_acceptance, pareto_sa
from src.instance import FlowShopInstance, generate_instance
from src.pareto import hypervolume_2d

MAX_ITERS: List[int] = [100, 200, 400, 800, 1600]
RUNS = 10
NADIR_FACTOR = 1.1
RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")


def _algo_rng(run: int, iter_index: int) -> RandomNumberGenerator:
    return RandomNumberGenerator(1000 + run * 100 + iter_index)


def _run_all_iters(
    instance: FlowShopInstance, run: int
) -> Dict[int, ParetoResult]:
    accept = constant_acceptance(0.1)
    results: Dict[int, ParetoResult] = {}
    for idx, max_iter in enumerate(MAX_ITERS):
        rng = _algo_rng(run, idx)
        results[max_iter] = pareto_sa(instance, rng, max_iter, accept)
    return results


def _nadir(results: Dict[int, ParetoResult]) -> Tuple[float, float]:
    z1 = max(p[0] for res in results.values() for p in res.front)
    z2 = max(p[1] for res in results.values() for p in res.front)
    return z1 * NADIR_FACTOR, z2 * NADIR_FACTOR


def make_scatter_plots(instance: FlowShopInstance, run: int) -> None:
    results = _run_all_iters(instance, run)
    for max_iter, res in results.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        px = [p[0] for p in res.population]
        py = [p[1] for p in res.population]
        fx = [p[0] for p in sorted(res.front)]
        fy = [p[1] for p in sorted(res.front)]
        ax.scatter(px, py, c="tab:blue", s=20, label="P (set)")
        ax.plot(fx, fy, "o-", c="tab:red", label="F (front)")
        ax.set_xlabel("Criterion 1: SumF")
        ax.set_ylabel("Criterion 2: Tmax")
        ax.set_title(f"Pareto SA, maxIter = {max_iter}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"task1_scatter_{max_iter}.png"), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for max_iter, res in results.items():
        front = sorted(res.front)
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            "o-",
            label=f"maxIter = {max_iter}",
        )
    ax.set_xlabel("Criterion 1: SumF")
    ax.set_ylabel("Criterion 2: Tmax")
    ax.set_title("Pareto fronts for all maxIter values")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task1_fronts_combined.png"), dpi=120)
    plt.close(fig)


def compute_hvi_table(instance: FlowShopInstance) -> Dict[int, float]:
    totals: Dict[int, float] = {m: 0.0 for m in MAX_ITERS}
    for run in range(RUNS):
        results = _run_all_iters(instance, run)
        reference = _nadir(results)
        for max_iter, res in results.items():
            totals[max_iter] += hypervolume_2d(res.front, reference)
    return {m: totals[m] / RUNS for m in MAX_ITERS}


def save_hvi_table(table: Dict[int, float]) -> None:
    path = os.path.join(RESULTS_DIR, "task1_hvi.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["maxIter", "mean_HVI"])
        for max_iter in MAX_ITERS:
            writer.writerow([max_iter, f"{table[max_iter]:.2f}"])


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance = generate_instance(n=50, seed=1)
    make_scatter_plots(instance, run=0)
    table = compute_hvi_table(instance)
    save_hvi_table(table)
    print("Task 1 - mean HVI over", RUNS, "runs (constant p=0.1, insert move)")
    print(f"{'maxIter':>8}  {'mean HVI':>14}")
    for max_iter in MAX_ITERS:
        print(f"{max_iter:>8}  {table[max_iter]:>14.2f}")
    print("\nPlots and task1_hvi.csv saved to results/")


if __name__ == "__main__":
    main()
