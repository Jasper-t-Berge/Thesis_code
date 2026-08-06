"""
Which topologies can produce which behaviours.

Sweeps every combination of the three topologies against the three target
t-graphs, running CMA-ES many times per cell, and draws the result as a
heatmap of success rate and average evaluations to solution.

CMA-ES is used because it is the most evaluation-efficient of the four
optimisers on this problem, which matters when the sweep runs it 9000 times.

A pair is called incompatible when the success rate is 0% across all trials.
With CMA-ES typically converging in a few hundred evaluations on a compatible
pair, zero successes over a thousand independent seeds is strong evidence that
no parameter set exists at all -- though strictly it shows only that none was
found for *this* target t-graph, not that the topology cannot produce that
kind of behaviour in any form.

The budget is deliberately small at 1000 evaluations per trial. On an
incompatible pair the optimiser would otherwise burn a large budget for
nothing, and on a compatible pair 1000 is already generous.

The search itself comes from `hysteron.optimize`, the same one the optimiser
benchmark uses, so both experiments run CMA-ES with identical settings. This
file supplies only the per-cell pieces: the geometry, the target, the cost
function and the starting points.
"""

import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import warnings                                               # noqa: E402

import numpy as np                                            # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

from hysteron import (                                        # noqa: E402
    HysteronValve,
    build_incidence_matrix,
    build_inequalities,
    config,
    cost_function_system,
    initialize_geometry,
    initialize_random_hysteron_valves,
    initialize_tgraph,
    straighten_parameters,
)
from hysteron.optimize import (                               # noqa: E402
    SIGMA0,
    SIGMA_FLOOR,
    STAGNATION_LIMIT,
    cma_search,
)

# --------------------------------------------------------------------------
# Sweep configuration
# --------------------------------------------------------------------------
TOPOLOGIES = ['series', 'series-parallel', 'wheatstone']
TGRAPHS    = ['Preisach', 'Avalanche', 'MPO']

NUM_TRIALS = 1000     # independent seeds per cell
MAX_FEVALS = 1000     # cost evaluation budget per trial

H_LO, H_HI = 0.1, 1.0
C_LO, C_HI = 0.5, 5.0

boundary_pressure_template = config.boundary_pressure_template


def _build_problem(topology, tgraph):
    """
    Assemble the geometry and inequalities for one cell of the sweep.

    Returns None if the pair cannot be set up at all, so the caller can mark
    the cell as not applicable rather than crash.
    """
    edges, N_valves, N_conds, N_nodes = initialize_geometry(topology=topology)
    N_edges = N_valves + N_conds
    inc = build_incidence_matrix(edges, N_edges, N_nodes)

    up_trans, down_trans = initialize_tgraph(tgraph=tgraph)
    if not up_trans and not down_trans:
        return None

    try:
        ineq_array = build_inequalities(N_valves, up_trans, down_trans)
    except ValueError as e:
        print(f"  -> Skipping: {e}")
        return None

    return edges, N_valves, N_conds, N_nodes, inc, ineq_array, up_trans, down_trans


def _run_one(seed, problem):
    """
    One CMA-ES trial on a single topology / t-graph pair.

    The search comes from `hysteron.optimize`; everything here is the
    per-cell wrapping around it.

    Returns `(success, best_cost, fevals)`.
    """
    edges, N_valves, N_conds, N_nodes, inc, ineq_array, up_trans, down_trans = problem

    rng = np.random.default_rng(seed)

    def fresh_x0():
        # Reseed the global RNG from this trial's generator, so the whole
        # sweep is reproducible from the single seed array built in
        # run_sweep. Everything downstream, including pycma's own stream,
        # follows from this one draw.
        sub_seed = int(rng.integers(0, 2**31))
        np.random.seed(sub_seed)
        conds_ = np.random.uniform(C_LO, C_HI, N_conds)
        _, h_ms, h_ps, C0s, dCs = initialize_random_hysteron_valves(
            N_valves, H_LO, H_HI, C_LO, C_HI)
        return np.concatenate((h_ms, h_ps, C0s, dCs, conds_))

    def reconstruct(p):
        p = np.array(p, dtype=float)
        v = HysteronValve(
            np.zeros(N_valves),
            p[:N_valves].copy(),
            p[N_valves:2 * N_valves].copy(),
            p[2 * N_valves:3 * N_valves].copy(),
            p[3 * N_valves:4 * N_valves].copy(),
        )
        c = p[4 * N_valves:].copy()
        p, v, c = straighten_parameters(p, v, c)
        return v, c

    def cost_fn(p):
        # An incompatible pair can drive the optimiser into networks that are
        # numerically degenerate, so anything non-finite is scored as very bad
        # rather than allowed to propagate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                v, c = reconstruct(p)
                val = cost_function_system(
                    ineq_array, v, c, boundary_pressure_template, inc)
                return float(val) if np.isfinite(val) else 1e6
            except Exception:
                return 1e6

    res = cma_search(cost_fn, fresh_x0, MAX_FEVALS,
                     sigma0=SIGMA0, sigma_floor=SIGMA_FLOOR,
                     stagnation_limit=STAGNATION_LIMIT)

    return res['success'], res['best_cost'], res['fevals']


def run_sweep():
    """
    Run every topology x t-graph cell.

    Returns `(success_rate, avg_evals)`, both shaped (topologies, tgraphs),
    with NaN marking a cell that could not be set up.
    """
    n_top = len(TOPOLOGIES)
    n_tg  = len(TGRAPHS)
    success_rate = np.full((n_top, n_tg), np.nan)
    avg_evals    = np.full((n_top, n_tg), np.nan)

    # All seeds drawn up front from one generator, so the entire sweep is
    # reproducible and every cell gets its own independent set.
    rng   = np.random.default_rng(42)
    seeds = rng.integers(1, 10**9, size=(n_top, n_tg, NUM_TRIALS))

    total_cells = n_top * n_tg
    cell = 0

    for i, topology in enumerate(TOPOLOGIES):
        for j, tgraph in enumerate(TGRAPHS):
            cell += 1
            print(f"\n[{cell}/{total_cells}] topology={topology}  tgraph={tgraph}")

            problem = _build_problem(topology, tgraph)
            if problem is None:
                print(f"  -> tgraph '{tgraph}' not defined for this topology, skipping.")
                continue

            successes      = 0
            all_costs      = []
            success_fevals = []

            for t in range(NUM_TRIALS):
                result, best_cost, fevals = _run_one(int(seeds[i, j, t]), problem)
                if result:
                    successes += 1
                    success_fevals.append(fevals)
                all_costs.append(best_cost)

                print(f"  trial {t + 1:>4}/{NUM_TRIALS}: "
                      f"{'SUCCESS' if result else 'fail  '}"
                      f"  best_cost={best_cost:>10.4f}"
                      f"  (running: {successes}/{t + 1})")

            success_rate[i, j] = successes / NUM_TRIALS
            if success_fevals:
                avg_evals[i, j] = np.mean(success_fevals)

            finite_costs = [c for c in all_costs if np.isfinite(c)]
            if finite_costs:
                print(f"\n  cost summary over {NUM_TRIALS} trials:")
                print(f"    min={min(finite_costs):.4f}  "
                      f"median={np.median(finite_costs):.4f}  "
                      f"max={max(finite_costs):.4f}")
            if success_fevals:
                print(f"  avg evals to solution (successful): {avg_evals[i, j]:.0f}")
            print(f"  => success rate: {100 * success_rate[i, j]:.1f}%")

    return success_rate, avg_evals


def plot_heatmap(success_rate, avg_evals):
    """Draw the sweep: success rate per cell, annotated with evaluation counts."""
    fig, ax = plt.subplots(figsize=(max(6, len(TGRAPHS) * 2.2),
                                    max(4, len(TOPOLOGIES) * 1.6)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='#2a2a2a')

    masked = np.ma.masked_invalid(success_rate * 100)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100,
                   aspect='auto', interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Success rate (%)', color='#e0e0e0', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#e0e0e0')

    ax.set_xticks(range(len(TGRAPHS)))
    ax.set_yticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels(TGRAPHS,    color='#e0e0e0', fontsize=12, fontweight='bold')
    ax.set_yticklabels(TOPOLOGIES, color='#e0e0e0', fontsize=12, fontweight='bold')
    ax.set_xlabel('T-graph type', color='#e0e0e0', fontsize=13, labelpad=10)
    ax.set_ylabel('Topology',     color='#e0e0e0', fontsize=13, labelpad=10)
    ax.tick_params(colors='#e0e0e0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    for i in range(len(TOPOLOGIES)):
        for j in range(len(TGRAPHS)):
            val = success_rate[i, j]
            if np.isnan(val):
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='#888')
                continue

            # Dark text on the green end, light text on the red end.
            textcolor = '#111' if val > 0.5 else '#eee'
            ax.text(j, i - 0.15, f'{val * 100:.1f}%', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=textcolor)
            ev = avg_evals[i, j]
            if not np.isnan(ev):
                ax.text(j, i + 0.25, f'({int(round(ev))} evals)',
                        ha='center', va='center', fontsize=8, color=textcolor)

    ax.set_title(
        f'CMA-ES  - success rate\n'
        f'({NUM_TRIALS} trials x {MAX_FEVALS:,} fevals each)',
        color='#e0e0e0', fontsize=14, fontweight='bold', pad=14)

    plt.tight_layout()
    plt.savefig('heatmap_cma.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nHeatmap saved to heatmap_cma.png")
    plt.show()


if __name__ == '__main__':
    success_rate, avg_evals = run_sweep()

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    header = f"{'Topology':<18}" + "".join(f"{tg:>14}" for tg in TGRAPHS)
    print(header)
    print("-" * len(header))
    for i, topology in enumerate(TOPOLOGIES):
        row = f"{topology:<18}"
        for j in range(len(TGRAPHS)):
            val = success_rate[i, j]
            row += f"{'N/A':>14}" if np.isnan(val) else f"{val * 100:>13.1f}%"
        print(row)

    plot_heatmap(success_rate, avg_evals)
