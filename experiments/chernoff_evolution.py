from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from RandomNumberGenerator import RandomNumberGenerator
from experiments.task2 import estimate_coefficients
from src.algorithms import constant_acceptance, scalarized_sa
from src.chernoff import draw_face, normalize_columns
from src.instance import FlowShopInstance, generate_instance
from src.schedule import criteria_four

Vector = Tuple[int, int, int, int]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")
MAX_ITER = 1600
FRAME_STEP = 32
SEED = 4242
FPS = 8


def collect_trajectory(
    instance: FlowShopInstance, coefficients: Tuple[float, float, float]
) -> List[Tuple[int, Vector]]:
    checkpoints = set(range(FRAME_STEP, MAX_ITER + 1, FRAME_STEP))
    checkpoints.add(1)
    frames: List[Tuple[int, Vector]] = []

    def on_step(it: int, best_x: List[int]) -> None:
        if it in checkpoints:
            frames.append((it, criteria_four(instance, best_x)))

    rng = RandomNumberGenerator(SEED)
    scalarized_sa(
        instance, rng, MAX_ITER, coefficients, constant_acceptance(0.1), on_step=on_step
    )
    return frames


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance = generate_instance(n=50, seed=1)
    coefficients = estimate_coefficients(instance)

    frames = collect_trajectory(instance, coefficients)
    iters = [it for it, _ in frames]
    vectors = [vec for _, vec in frames]
    normalized = normalize_columns(vectors)

    fig, ax = plt.subplots(figsize=(4.0, 6.0))

    def render(index: int) -> None:
        ax.clear()
        draw_face(ax, normalized[index], f"xbest after {iters[index]} iterations")

    animation = FuncAnimation(fig, render, frames=len(frames))
    path = os.path.join(RESULTS_DIR, "chernoff_evolution.gif")
    animation.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)

    print(f"Saved {len(frames)} frames ({iters[0]}..{iters[-1]} iterations) to {path}")
    print(f"first xbest (SumF, Tmax, Lmax, SumL) = {vectors[0]}")
    print(f"last  xbest (SumF, Tmax, Lmax, SumL) = {vectors[-1]}")


if __name__ == "__main__":
    main()
