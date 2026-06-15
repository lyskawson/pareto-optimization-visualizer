from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, ...]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    no_worse = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return no_worse and strictly_better


def pareto_front(points: Sequence[Point]) -> List[Point]:
    front: List[Point] = list(points)
    result: List[Point] = []
    for a in front:
        if not any(b != a and dominates(b, a) for b in front):
            result.append(a)
    return result


def hypervolume_2d(front: Sequence[Tuple[float, float]], reference: Tuple[float, float]) -> float:
    rx, ry = reference
    relevant = sorted(
        (p for p in front if p[0] <= rx and p[1] <= ry), key=lambda p: p[0]
    )
    area = 0.0
    prev_x = rx
    for x, y in reversed(relevant):
        area += (prev_x - x) * (ry - y)
        prev_x = x
    return area
