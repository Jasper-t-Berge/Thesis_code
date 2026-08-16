# Inverse Design of Fluidic Hysteron Networks

Simulation and inverse-design code for networks of fluidic hysteron valves,
and a comparison of five optimisation methods for solving the inverse design
problem they pose.

## Background

A **fluidic hysteron valve** is a bistable element: it semi-closes once the
pressure drop across it rises above a closing threshold, and re-opens only
once the drop falls back below a lower opening threshold. The gap between the
two thresholds is a hysteresis loop, and it gives the valve a rudimentary
memory of what the pressure did in the past.

Wire several valves into a network and they interact, because one valve
switching changes its conductance and so redistributes the flow through all
the others. Depending on the wiring, that coupling can be negative (a valve
closing makes its neighbours harder to close) or positive (it makes them
easier), and mixing the two produces genuinely complicated behaviour.

Sweeping the source pressure up and down therefore drives the network through
a sequence of global states. That sequence is captured by a **transition
graph** (t-graph): a state machine whose nodes are the open/closed states of
all valves and whose edges are the transitions between them. Some t-graphs are
computationally interesting — a multi-periodic orbit, for instance, returns to
its starting state only after *n* pressure pulses, which lets the network count
modulo *n*.

The **inverse design problem** is to go the other way: from a desired t-graph
back to the valve parameters that produce it. This is hard because the mapping
from parameters to behaviour is highly nonlinear — a small change to one
threshold can restructure the whole t-graph — and because "produces the right
t-graph" is a yes/no property, which gives an optimiser nothing to descend.

The approach taken here turns each transition in the target t-graph into a set
of inequalities between switching thresholds, and sums how badly they are
violated into a single continuous cost. That cost is zero exactly when the
network produces the target behaviour, and grows smoothly with the size of the
violations, so an optimiser can tell a nearly correct network from a badly
wrong one.

The repository answers three questions:

1. **Which optimiser solves this best?** Gradient descent, Adam, CMA-ES,
   Bayesian TPE and a hybrid CMA-ES + linear-programming method are
   benchmarked on a fixed topology and target.
2. **Which topologies support which behaviours?** The best optimiser is then
   used to map three topologies against three target behaviours.
3. **How does this scale?** The methods are swept over networks from 4 up to
   60 valves, to see which of them survive the growth in dimension.

## Layout

```
demo.py                       solve one network and draw the result

hysteron/                     the shared library
  config.py                   boundary conditions, cost margin
  network.py                  valves, topologies, the Kirchhoff pressure solve
  tgraph.py                   target behaviours, forward t-graph extraction
  inequalities.py             design inequalities and the cost function
  parameters.py               parameter layout, initialisation, kicks
  plotting.py                 t-graph visualisation

compare_optimizers/           which optimiser is best
  problem.py                  the shared problem definition
  solver_gradient_descent.py
  solver_adam.py
  solver_cma.py
  solver_bayesian.py
  solver_hybrid.py            CMA-ES over conductances, LP over thresholds
  run_comparison.py           runs all five and plots the comparison
  scaled_problem.py           a Wheatstone ladder at any size, for scaling runs
  run_scaling.py              sweeps the network size, 4 to 60 valves

topology_sweep/               which topology supports what
  run_heatmap.py              sweeps topology x t-graph, draws the heatmap
```

Three topologies (`series`, `series-parallel`, `wheatstone`) and three target
t-graphs (`Preisach`, `Avalanche`, `MPO`) are defined in `initialize_geometry`
and `initialize_tgraph`. Every topology uses 4 hysteron valves plus one fixed
conductor as its 5th edge, which keeps the search space at 17 dimensions in
all cases and makes the three directly comparable.

## The hybrid solver

Four of the five methods treat all 17 parameters as one opaque search space.
Half of that space does not need to be searched at all.

Every design inequality compares two switching thresholds, and a threshold is

```
T(i, s) = h_i(s) / dP(i, s)
```

where `h_i(s)` is one of valve *i*'s two intrinsic pressures and `dP(i, s)` is
the fraction of the source pressure dropped across edge *i* in state *s*. The
pressure drop falls out of the Kirchhoff solve, whose only inputs are the
conductances and the state — the threshold parameters never enter it.

So fix the 9 conductances and every threshold becomes a known constant times
one of the 8 remaining unknowns. Each cost term `max(0, T_a - T_b + eps)` is
then a hinge on a linear function of those unknowns, the total cost is convex
and piecewise linear in them, and its exact minimum is the optimum of a small
linear program. CMA-ES is left searching only conductance space — 9 dimensions
instead of 17 — and for each candidate the LP returns the best cost achievable
over *all* thresholds.

The LP is the epigraph form of the cost, solved with HiGHS:

```
minimise    sum_k slack_k
subject to  c_small,k * h_small,k - c_large,k * h_large,k + eps <= slack_k
            h+_i - h-_i >= 0.1
            slack_k >= 0,  h-_i >= 0.05
```

Its feasible box is exactly the region `straighten_parameters` projects into,
so the hybrid is allowed no thresholds the other solvers are denied.

### Results

Ten seeds on the Wheatstone/MPO benchmark, 10,000 evaluations each. Medians
are over successful runs. Every success was re-verified by rebuilding the
network from the returned parameters and re-scoring it:

| method   | success | evaluations | wall-clock |
|----------|---------|-------------|------------|
| GD       | 0/10    | —           | —          |
| Adam     | 9/10    | 1782        | 0.95 s     |
| CMA      | 8/10    | 432         | 0.34 s     |
| Bayesian | 10/10   | 1490        | 19.47 s    |
| Hybrid   | 9/10    | **24**      | **0.03 s** |

Ten trials is enough to see the shape and not enough to separate 8/10 from
9/10; use `compare(num_trials=1000)` for numbers worth quoting.

Two caveats on reading that table. A hybrid evaluation costs about 3.9x a
plain one, because it solves an LP on top of the same pressure solves, so the
evaluation column flatters it and only the wall-clock column prices it
honestly — though it wins on both. And the LP separates thresholds by 1e-5
rather than `config.EPSILON`'s 1e-9, because 1e-9 sits below HiGHS's own
feasibility tolerance and would let it call a constraint satisfied when it was
not. The coarser margin errs only towards rejecting marginal solutions, never
towards accepting broken ones, and costs nothing because the cost is
homogeneous in the thresholds.

The more interesting result is a by-product: **about 63% of purely random
conductance vectors already solve the benchmark** once the LP picks thresholds
optimally, which is why the median is two generations. The difficulty in this
problem was almost entirely in the threshold parameters rather than the
conductances.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+, and `numpy`, `matplotlib`, `networkx` and `cma` (pycma).

Nothing else is needed. Every script puts the project root on `sys.path`
itself, so the files run directly from an IDE's run button or the command
line. If you would rather have `hysteron` importable from anywhere, there is
a `pyproject.toml`, so

```bash
pip install -e .
```

also works. Both routes are supported; neither is required.

## Running

Start here. The demo solves one network, checks the answer by sweeping the
pressure over it, and draws the resulting t-graph. It takes a few seconds:

```bash
python demo.py
```

That last check is the meaningful one. The optimiser only ever minimises the
inequality cost, which is a proxy; simulating the solved network and reading
its t-graph back off is what confirms the intended behaviour is actually
there.

Any single solver, for one run with a convergence plot:

```bash
python compare_optimizers/solver_cma.py
```

The full optimiser benchmark. The default is 1000 trials per method, which
takes a few hours; edit the `compare(num_trials=...)` call at the bottom of the
file for a quicker look:

```bash
python compare_optimizers/run_comparison.py
```

The topology sweep, 1000 trials per cell across 9 cells:

```bash
python topology_sweep/run_heatmap.py
```

Both runners save their figures as PNG files in the working directory.

The scaling sweep, which runs Adam, CMA and the hybrid over a Wheatstone
ladder of increasing size. The arguments are the ladder sizes (each cell is 4
valves), the seeds per size, the pulses used to build the target, and the
evaluation budget:

```bash
python compare_optimizers/run_scaling.py 1,2,4,8 10 40 10000
```

It prints a row per size and method as each finishes. The larger sizes are
slow — 32 valves is roughly half an hour per method — so start with `1,2`.

## Picking this up for the first time

Suggested reading order. It follows the physics rather than the file listing:

1. **`demo.py`** — run it. It exercises the whole pipeline in a few seconds
   and shows you what the project produces.
2. **`hysteron/network.py`** — how a valve is represented, how the three
   topologies are wired, and how the pressure solve works. Everything else
   sits on top of this.
3. **`hysteron/inequalities.py`** — the conceptual core: how a desired
   behaviour becomes a number an optimiser can minimise. If you only read one
   file properly, read this one.
4. **`compare_optimizers/solver_cma.py`** — the best-performing optimiser,
   and the shortest of the four to follow.

### Adding a new topology

Add a branch to `initialize_geometry` in `hysteron/network.py`. Three
conventions the rest of the code depends on:

- node 0 is the sink, node 1 is the source, nodes 2+ are internal;
- edges `0 .. N_valves-1` are valves, everything after is a fixed conductor;
- keep 4 valves and 1 conductor if you want your results to be comparable with
  the existing ones, since that is what fixes the parameter space at 17.

### Adding a new target behaviour

Add a branch to `initialize_tgraph` in `hysteron/tgraph.py`, listing the
up- and down-transitions that define it. Only the transitions you care about
need listing; the optimiser fills in the rest however it likes.

Two limitations to know about.

The inequality generator assumes up-transitions only ever close valves and
down-transitions only ever open them. All three existing targets satisfy that.
A target that mixes directions within a single transition will generate wrong
constraints, not an error, so check yours before trusting a result.

It is also **stricter than the physics for cascades**. For a multi-flip
transition, `inequalities_up_transition` loops over every flipped bit as the
critical one and demands each of them be the first to flip out of the initial
state. Physically only one valve triggers and the rest follow, so a network
that genuinely produces the avalanche still scores a nonzero cost on its own
recorded behaviour. `Preisach` and `MPO` are entirely single-flip and never hit
this; `Avalanche`'s `3 -> 15` does. The cheap check on any new target is to
draw a random network, read its behaviour back off with the forward solver, and
confirm it scores zero on the target it just produced.

### Adding a new optimiser

Copy any solver in `compare_optimizers/`. The contract is small: import
`problem`, expose `run(seed, ..., show_plot=...)`, and return a list of dicts
with `success`, `final_cost`, `completion_index`, `cost_arr` and `params`.
Register it in the `methods` dict in `run_comparison.py` along with how many
cost evaluations one recorded step costs, and it joins the benchmark.

`params` is the one that matters if you want to *use* a solver rather than
benchmark it: it is the parameter vector that solves the design problem. The
rest only describe how the search got there. `demo.py` shows the pattern —
solve, take `params`, then simulate the resulting network to check it really
behaves as intended.

## Possible extensions

Directions the project could be taken next:

- **Richer cascade constraints.** When several valves flip together, the
  generator constrains the co-flipping valves against each other. Also
  constraining the valves that stay put, in the intermediate state the cascade
  passes through, would tighten the avalanche and MPO targets. Worth checking
  what it does to their success rates.
- **Larger networks.** `compare_optimizers/scaled_problem.py` now reaches 60
  valves, but three things still bound it. States are packed bit patterns
  stored in numpy integer arrays, so the all-closed state `2**N - 1` overflows;
  `inequalities.py` uses `int64`, which caps the system at 62 valves, and going
  past that means replacing the packed-integer encoding outright. The pressure
  solve still uses an explicit `np.linalg.inv` where `np.linalg.solve` would be
  better conditioned. And the t-graph explorer enumerates all `2**N` states, so
  it is unusable above about 20 valves — the scaling code sidesteps it by
  following a pressure sweep instead, which visits only O(N) states.
- **Explicit random generators.** Randomness currently runs through numpy's
  global state. Threading `numpy.random.Generator` objects through instead
  would make each component's stream independent, which matters as soon as
  anything runs in parallel.
- **Topologies beyond the three here.** The sweep framework takes any topology
  and any target, so extending either list is mostly a matter of writing down
  the edges or the transitions.

## Verifying a result

Two checks are worth making on anything a solver hands back, for two different
reasons.

**The cost is only a proxy.** Zero cost means every design inequality holds,
which is what the target t-graph was translated into — but the thing you
actually want is the behaviour. Simulating the solved network and reading its
t-graph back off is what confirms the behaviour is there. `demo.py` shows the
pattern: solve, take `params`, then sweep the pressure over the result.

**A reported cost belongs to a specific parameter vector.** `cma_search`
guarantees that `best_cost` and `best_params` describe the same network, so
re-scoring `params` with `cost_function_system` should reproduce the reported
cost exactly. `run_scaling.py` does this on every claimed success, and it is
cheap enough to be worth doing whenever a number is going into a write-up.

That second guarantee has not always held. Until the fix in this branch,
`best_cost_overall` was initialised to `np.inf` while `cost_arr` — a running
minimum that `success` and `best_cost` are both read off — already contained
the starting point's cost. A run that began on a solution therefore reported
zero while generation 1 quietly overwrote `best_params` with a worse point.
Results generated before the fix can report a cost their parameters do not
achieve; the tell-tale is `success=True` with `completion_index == -1` and the
full evaluation budget spent. Re-scoring `params` catches it.

## License

Released under the MIT License; see [LICENSE](LICENSE).
