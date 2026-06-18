from __future__ import annotations

from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

FEATURE_NAMES = [
    "face height (SumF)",
    "mouth curvature (Tmax)",
    "eye size (Lmax)",
    "eyebrow slope (SumL)",
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
    face_h = 1.0 + 0.55 * _feature(values, 0)
    ax.add_patch(
        Ellipse((0.0, 0.0), width=2.0, height=2.0 * face_h, fill=False, lw=2.0)
    )

    eye_r = 0.12 + 0.14 * _feature(values, 2)
    eye_y = 0.30 * face_h
    for sign in (-1.0, 1.0):
        ex = sign * 0.42
        ax.add_patch(
            Ellipse((ex, eye_y), width=2 * eye_r, height=2 * eye_r, fill=False, lw=1.5)
        )
        ax.add_patch(Ellipse((ex, eye_y), width=0.08, height=0.08, color="black"))

    brow_drop = (_feature(values, 3) - 0.5) * 0.35
    brow_y = eye_y + eye_r + 0.10
    for sign in (-1.0, 1.0):
        outer_x = sign * 0.60
        inner_x = sign * 0.24
        ax.plot(
            [outer_x, inner_x],
            [brow_y, brow_y - brow_drop],
            color="black",
            lw=2.0,
        )

    nose_y = eye_y - 0.05
    ax.plot([0.0, 0.0], [nose_y, nose_y - 0.35], color="black", lw=1.5)

    smile = (0.5 - _feature(values, 1)) * 2.0
    mouth_y = -0.45 * face_h
    xs = [(-0.4 + 0.8 * i / 40) for i in range(41)]
    ys = [mouth_y + 0.25 * smile * (x / 0.4) ** 2 for x in xs]
    ax.plot(xs, ys, color="black", lw=2.0)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-2.0, 2.0)
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
