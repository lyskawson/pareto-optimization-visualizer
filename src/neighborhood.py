from __future__ import annotations

from typing import List

from RandomNumberGenerator import RandomNumberGenerator


def random_permutation(n: int, rng: RandomNumberGenerator) -> List[int]:
    pi = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.nextInt(0, i)
        pi[i], pi[j] = pi[j], pi[i]
    return pi


def swap_neighbor(pi: List[int], rng: RandomNumberGenerator) -> List[int]:
    n = len(pi)
    i = rng.nextInt(0, n - 1)
    j = rng.nextInt(0, n - 1)
    neighbor = list(pi)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def insert_neighbor(pi: List[int], rng: RandomNumberGenerator) -> List[int]:
    n = len(pi)
    i = rng.nextInt(0, n - 1)
    j = rng.nextInt(0, n - 1)
    neighbor = list(pi)
    job = neighbor.pop(i)
    neighbor.insert(j, job)
    return neighbor


MOVES = {"swap": swap_neighbor, "insert": insert_neighbor}


def random_neighbor(
    pi: List[int], rng: RandomNumberGenerator, move: str = "insert"
) -> List[int]:
    return MOVES[move](pi, rng)
