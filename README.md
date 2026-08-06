# Inverse Design of Fluidic Hysteron Networks

Simulation and inverse-design code for networks of fluidic hysteron valves,
and a comparison of four optimisation methods for solving the inverse design
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

The repository answers two questions:

1. **Which optimiser solves this best?** Gradient descent, Adam, CMA-ES and
   Bayesian TPE are benchmarked on a fixed topology and target.
2. **Which topologies support which behaviours?** The best optimiser is then
   used to map three topologies against three target behaviours.

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
  run_comparison.py           runs all four and plots the comparison

topology_sweep/               which topology supports what
  run_heatmap.py              sweeps topology x t-graph, draws the heatmap
```

Three topologies (`series`, `series-parallel`, `wheatstone`) and three target
t-graphs (`Preisach`, `Avalanche`, `MPO`) are defined in `initialize_geometry`
and `initialize_tgraph`. Every topology uses 4 hysteron valves plus one fixed
conductor as its 5th edge, which keeps the search space at 17 dimensions in
all cases and makes the three directly comparable.

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

One limitation to know about: the inequality generator assumes up-transitions
only ever close valves and down-transitions only ever open them. All three
existing targets satisfy that. A target that mixes directions within a single
transition will generate wrong constraints, not an error, so check yours
before trusting a result.

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
- **Larger networks.** Everything here is built around 4 valves and 1
  conductor. Scaling up mainly means revisiting the pressure solve, where
  `np.linalg.solve` would be better conditioned than the explicit inverse used
  now, and the t-graph explorer, which enumerates all 2^N states.
- **Explicit random generators.** Randomness currently runs through numpy's
  global state. Threading `numpy.random.Generator` objects through instead
  would make each component's stream independent, which matters as soon as
  anything runs in parallel.
- **Topologies beyond the three here.** The sweep framework takes any topology
  and any target, so extending either list is mostly a matter of writing down
  the edges or the transitions.

## License

Released under the MIT License; see [LICENSE](LICENSE).
