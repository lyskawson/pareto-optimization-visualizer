from __future__ import annotations

from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, Polygon

FEATURE_NAMES = [
    "face height (SumF)",
    "mouth curvature (Tmax)",
    "eye size and gaze (Lmax)",
    "eyebrow angle (SumL)",
]


def normalize_columns(
    solutions: Sequence[Sequence[float]],
) -> List[List[float]]:
    k = len(solutions[0])
    normalized = [[0.0] * k for _ in solutions]
    for c in range(k):
        column = [float(row[c]) for row in solutions]
        lo, hi = min(column), max(column)
        span = hi - lo
        for r, value in enumerate(column):
            normalized[r][c] = 0.0 if span == 0 else (value - lo) / span
    return normalized


def _feature(values: Sequence[float], index: int) -> float:
    return values[index] if index < len(values) else 0.5


def draw_face(ax: Axes, values: Sequence[float], title: str) -> None:
    half_h = 0.9 + 0.45 * _feature(values, 0)
    half_w = 0.9
    ax.add_patch(
        Ellipse((0.0, 0.0), width=2 * half_w, height=2 * half_h, fill=False, lw=2.0)
    )

    eye_y = 0.32 * half_h
    eye_x = 0.38
    eye_r = 0.13 + 0.12 * _feature(values, 2)
    pupil_dx = 0.5 * eye_r * _feature(values, 2)
    for sign in (-1.0, 1.0):
        ex = sign * eye_x
        ax.add_patch(Circle((ex, eye_y), eye_r, fill=False, lw=1.5))
        ax.add_patch(Circle((ex - sign * pupil_dx, eye_y), 0.045, color="black"))

    brow_angle = 40.0 * _feature(values, 3)
    brow_y = eye_y + eye_r + 0.14
    for sign in (-1.0, 1.0):
        ax.add_patch(
            Ellipse(
                (sign * eye_x, brow_y),
                width=0.40,
                height=0.05,
                angle=-sign * brow_angle,
                color="black",
            )
        )

    nose_top = eye_y - 0.02
    nose_base = eye_y - 0.42
    ax.add_patch(
        Polygon(
            [(0.0, nose_top), (-0.12, nose_base), (0.12, nose_base)],
            closed=False,
            fill=False,
            ec="black",
            lw=1.5,
        )
    )

    smile = (0.5 - _feature(values, 1)) * 2.0
    mouth_y = -0.45 * half_h
    xs = [(-0.34 + 0.68 * i / 40) for i in range(41)]
    ys = [mouth_y + 0.22 * smile * (x / 0.34) ** 2 for x in xs]
    ax.plot(xs, ys, color="black", lw=2.0)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def draw_faces(
    solutions: Sequence[Sequence[float]],
    labels: Sequence[str],
    path: str,
    suptitle: str = "Chernoff faces",
) -> None:
    normalized = normalize_columns(solutions)
    fig, axes = plt.subplots(1, len(solutions), figsize=(3.0 * len(solutions), 4.0))
    if len(solutions) == 1:
        axes = [axes]
    for ax, values, label in zip(axes, normalized, labels):
        draw_face(ax, values, label)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)
