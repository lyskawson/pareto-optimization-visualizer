from __future__ import annotations

from typing import List, Sequence, Tuple

from src.instance import FlowShopInstance, MACHINES


def completion_matrix(
    instance: FlowShopInstance, pi: Sequence[int]
) -> List[List[int]]:
    p = instance.p
    n = instance.n
    c = [[0] * n for _ in range(MACHINES)]
    for k, job in enumerate(pi):
        for i in range(MACHINES):
            tech_prev = c[i - 1][job] if i > 0 else 0
            machine_prev = c[i][pi[k - 1]] if k > 0 else 0
            c[i][job] = max(tech_prev, machine_prev) + p[i][job]
    return c


def last_machine_completion(
    instance: FlowShopInstance, pi: Sequence[int]
) -> List[int]:
    return completion_matrix(instance, pi)[MACHINES - 1]


def total_flowtime(cm: Sequence[int]) -> int:
    return sum(cm)


def max_tardiness(cm: Sequence[int], d: Sequence[int]) -> int:
    return max(max(cm[j] - d[j], 0) for j in range(len(cm)))


def max_lateness(cm: Sequence[int], d: Sequence[int]) -> int:
    return max(cm[j] - d[j] for j in range(len(cm)))


def total_lateness(cm: Sequence[int], d: Sequence[int]) -> int:
    return sum(cm[j] - d[j] for j in range(len(cm)))


def criteria_two(instance: FlowShopInstance, pi: Sequence[int]) -> Tuple[int, int]:
    cm = last_machine_completion(instance, pi)
    return total_flowtime(cm), max_tardiness(cm, instance.d)


def criteria_three(
    instance: FlowShopInstance, pi: Sequence[int]
) -> Tuple[int, int, int]:
    cm = last_machine_completion(instance, pi)
    return (
        total_flowtime(cm),
        max_tardiness(cm, instance.d),
        max_lateness(cm, instance.d),
    )


def criteria_four(
    instance: FlowShopInstance, pi: Sequence[int]
) -> Tuple[int, int, int, int]:
    cm = last_machine_completion(instance, pi)
    return (
        total_flowtime(cm),
        max_tardiness(cm, instance.d),
        max_lateness(cm, instance.d),
        total_lateness(cm, instance.d),
    )
