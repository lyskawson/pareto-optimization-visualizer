from __future__ import annotations

import math
from typing import List, Sequence, Tuple

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

RADAR_LABELS = ["SumF", "Tmax", "Lmax", "SumL"]


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


def _radar_point(
    center: Tuple[float, float], radius: float, angle_deg: float, value: float
) -> Tuple[float, float]:
    angle = math.radians(angle_deg)
    return (
        center[0] + radius * value * math.cos(angle),
        center[1] + radius * value * math.sin(angle),
    )


def draw_radar(
    ax: Axes,
    values: Sequence[float],
    center: Tuple[float, float],
    radius: float,
) -> None:
    k = len(values)
    angles = [90.0 - i * 360.0 / k for i in range(k)]
    for angle, label in zip(angles, RADAR_LABELS):
        rim = _radar_point(center, radius, angle, 1.0)
        ax.plot([center[0], rim[0]], [center[1], rim[1]], color="0.8", lw=0.8)
        text = _radar_point(center, radius * 1.18, angle, 1.0)
        ax.text(text[0], text[1], label, ha="center", va="center", fontsize=6, color="0.4")
    ax.add_patch(Circle(center, radius, fill=False, ec="0.85", lw=0.8))
    points = [
        _radar_point(center, radius, angle, value)
        for angle, value in zip(angles, values)
    ]
    ax.add_patch(
        Polygon(points, closed=True, facecolor="tab:blue", alpha=0.25, ec="tab:blue", lw=1.5)
    )
    for px, py in points:
        ax.add_patch(Circle((px, py), 0.03, color="tab:blue"))


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

    draw_radar(ax, values, center=(0.0, -2.7), radius=0.85)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-3.9, 1.7)
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
    fig, axes = plt.subplots(1, len(solutions), figsize=(3.0 * len(solutions), 6.0))
    if len(solutions) == 1:
        axes = [axes]
    for ax, values, label in zip(axes, normalized, labels):
        draw_face(ax, values, label)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)
