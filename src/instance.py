from __future__ import annotations

from dataclasses import dataclass
from typing import List

from RandomNumberGenerator import RandomNumberGenerator

MACHINES = 3


@dataclass(frozen=True)
class FlowShopInstance:
    n: int
    seed: int
    p: List[List[int]]
    d: List[int]


def generate_instance(n: int = 50, seed: int = 1) -> FlowShopInstance:
    rng = RandomNumberGenerator(seed)
    p = [[0] * n for _ in range(MACHINES)]
    total = 0
    for i in range(MACHINES):
        for j in range(n):
            value = rng.nextInt(1, 99)
            p[i][j] = value
            total += value
    low = total // 4
    high = total // 2
    d = [rng.nextInt(low, high) for _ in range(n)]
    return FlowShopInstance(n=n, seed=seed, p=p, d=d)
