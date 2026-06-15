# Multi-criteria Permutation Flow Shop (m = 3) — Lab 7, Tasks 1 & 2

Multi-criteria optimization of the permutation flow shop problem with three machines
(`problemy.pdf`, problem 2.8 / chapter 6). The repository implements **Task 1** (Pareto
Simulated Annealing + Hypervolume Indicator) and **Task 2** (scalarized Simulated Annealing)
from `zadania.pdf`.

## Problem

`n` jobs are processed on `m = 3` machines in the same job order on every machine (the
permutation `pi` is the decision variable). `p[i][j]` is the processing time of job `j` on
machine `i`. Completion times follow the recurrence

```
C[i][pi[k]] = max(C[i-1][pi[k]], C[i][pi[k-1]]) + p[i][pi[k]]
```

with `C[-1][...] = 0` and `C[i][pi[-1]] = 0`. Let `Cm[j] = C[2][j]` be the completion time of
job `j` on the last machine.

Each job has a due date `d[j]` defined in **input order** (job index `j`), not by execution
position. The criteria considered (all minimized) are:

- **Criterion 2 — Total flowtime**: `SumF = sum_j Cm[j]`
- **Criterion 3 — Max tardiness**: `Tmax = max_j max(Cm[j] - d[j], 0)`
- **Criterion 5 — Max lateness**: `Lmax = max_j (Cm[j] - d[j])` (may be negative)

Task 1 uses criteria `(2, 3) = (SumF, Tmax)`. Task 2 uses `(2, 3, 5) = (SumF, Tmax, Lmax)`.

Note that `Tmax = max_j max(0, Cm[j] - d[j]) = max(0, max_j (Cm[j] - d[j])) = max(0, Lmax)`.
So criteria 3 and 5 **coincide exactly whenever at least one job is tardy** (`Lmax >= 0`), and
they only diverge when every job finishes on time (`Lmax < 0`), where `Tmax` clamps to `0` while
`Lmax` stays negative. With the tightly-drawn due dates of this generator, `n = 50` random and
early-search schedules almost always have a tardy job, so `Tmax = Lmax` there; the two separate
only once the search finds schedules that meet every deadline.

## Instance generation

Instances are generated exactly as in §6 of `problemy.pdf`, using only the provided
`RandomNumberGenerator`. The order of `nextInt` calls is load-bearing for reproducibility:

```
init(seed)
S = 0
for i in 0..2:            # machine (outer loop)
    for j in 0..n-1:     # job (inner loop)
        p[i][j] = nextInt(1, 99)
        S += p[i][j]
for j in 0..n-1:
    d[j] = nextInt(floor(S/4), floor(S/2))
```

`p` is filled **machine-major** (3n calls), then the `n` due dates are drawn. Changing this
order changes the instance, so it must not be altered.

Two independent generator instances are used: one seeded for **instance generation**
(`generate_instance(seed=...)`) and a separate one for **algorithm stochasticity** (initial
permutation, neighbor selection, acceptance). This keeps instances reproducible independently
of the search.

## Layout

```
src/instance.py        # §6 generation, FlowShopInstance dataclass
src/schedule.py        # completion matrix + the three criteria
src/pareto.py          # dominance, front extraction, 2D HVI
src/neighborhood.py    # swap / insert moves, random permutation
src/algorithms.py      # pareto_sa (Task 1), scalarized_sa (Task 2)
experiments/task1.py   # scatter plots + HVI table
experiments/task2.py   # s(xbest) table
experiments/verify.py  # hand-trace + sanity checks
results/               # generated plots and tables
```

## Install & run

```
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
python -m experiments.verify             # sanity checks (no dependencies)
python -m experiments.task1              # plots + HVI table -> results/
python -m experiments.task2              # s(xbest) table   -> results/
```

Run all commands from the repository root so that the `src` package resolves.

The fixed instance is `n = 50, seed = 1`. Algorithm seeds are derived deterministically per
run/maxIter inside the experiments, so re-running reproduces identical numbers and plots.

## Task 1 — Pareto SA + HVI

`pareto_sa` follows the pseudocode of §3: start from a random permutation, repeatedly draw a
random neighbor, accept it if it dominates the current solution, otherwise accept with
probability `p(it)`; every accepted solution is added to `P`. The Pareto front `F` is extracted
by removing dominated points from `P`.

- Acceptance probability: constant (`p(it) = 0.1`, default) or geometric (`p(it) = 0.995**it`),
  selectable via `constant_acceptance` / `geometric_acceptance`.
- Neighborhood: `swap` and `insert` are both implemented; default is `insert`.
- `maxIter in [100, 200, 400, 800, 1600]`.

Outputs (in `results/`):

- `task1_scatter_<maxIter>.png` — full set `P` (blue) with front `F` highlighted (red), one per
  `maxIter` (X = `SumF`, Y = `Tmax`).
- `task1_fronts_combined.png` — the five fronts overlaid.
- `task1_hvi.csv` — `maxIter -> mean HVI`, also printed to stdout.

**HVI.** For each run, the nadir point `Z = (z1, z2)` is the worst (largest) criterion-1 and
criterion-2 value across that run's five fronts, multiplied by `NADIR_FACTOR = 1.1`. The 2D HVI
is the dominated area between the front and `Z` (sort by x ascending, sum the staircase
rectangles). Because the search is stochastic, the whole procedure is repeated `RUNS = 10`
times and the **mean HVI** per `maxIter` is reported (nadir recomputed per run). The plots come
from one fixed representative run (`run = 0`).

## Task 2 — Scalarized SA

`scalarized_sa` follows §4: the three criteria are reduced to a scalar
`s(x) = c1*SumF + c2*Tmax + c3*Lmax`. Acceptance/best-tracking follow the pseudocode.

**Scaling decision.** `SumF` is a sum over `n` jobs (order of tens of thousands), while `Tmax`
and `Lmax` are single maxima (order of hundreds to low thousands). To make the contributions
comparable, the coefficients are set to the inverse of each criterion's estimated scale:
`SCALE_SAMPLES = 50` random permutations are drawn (with a dedicated seeded generator),
`scale_i` is the mean of `|criterion_i|` over those samples (floored at 1.0), and
`c_i = 1 / scale_i`. This normalizes every criterion to a comparable `~1` range with equal
weight on the normalized values. The coefficient vector is exposed as the `coefficients`
parameter of `scalarized_sa`, so it can be overridden (e.g. to study the `c1 = 5*c2` style
weighting suggested in the lab) without touching the algorithm.

The normalization balances the criteria at the random-sample estimation point (`s ~ 3` for a
random schedule). After optimization the split is naturally dominated by the `SumF` term offset
by a negative `Lmax` term: a good schedule meets every deadline (`Tmax -> 0`) with slack, so
`Lmax` becomes negative and the optimized `s(xbest)` is small and can dip below zero. This is
expected, not a bug.

Outputs (in `results/`):

- `task2_scalar.csv` — the coefficient vector plus `maxIter -> mean s(xbest)` over `RUNS = 10`
  runs, also printed to stdout.

Lower `s(xbest)` is better; it decreases as `maxIter` grows.

## Reproducibility checks

`experiments/verify.py` prints a small `n = 5` instance and hand-checks the first completion
times against the recurrence, checks dominance and front extraction on a tiny point set,
checks 2D HVI on the single-point (rectangle) and two-point cases, and confirms that two
instances generated with the same seed are identical.

## Out of scope

Task 3 and the seven named visualization methods (bar plots, value paths, dotplots, star
coordinates, star coordinates with segments, spider charts, Chernoff faces) are intentionally
not implemented.
