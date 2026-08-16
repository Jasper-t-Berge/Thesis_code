"""
Benchmark all five optimisers against each other.

Runs every method over the same set of seeds on the same problem, then plots
three comparisons:

  1. success rate -- how often each method reached zero cost at all
  2. cost evaluations to converge
  3. wall-clock time to converge

and prints the medians, interquartile ranges and bootstrap confidence
intervals behind them.

Both speed measures are reported because neither is sufficient alone. The TPE
solver spends real time building its surrogate between evaluations, so
counting evaluations flatters it; wall-clock time captures that overhead but
depends on the machine, so only large differences in it mean much. The same
caveat applies with more force to the hybrid solver, whose every evaluation
carries a linear program: on the evaluation axis it is being given a discount
that only the wall-clock axis charges it for.

Each solver exposes a `run(seed, ...)` that executes its complete optimisation
loop, so what is benchmarked here is exactly what runs when a solver is
invoked on its own.

Running the full 1000 trials per method takes a few hours. Pass a smaller
`num_trials` for a quick look.
"""

import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import time                                                   # noqa: E402

import numpy as np                                            # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.lines import Line2D                           # noqa: E402

import problem                                                # noqa: E402
import solver_gradient_descent as gd_mod                      # noqa: E402
import solver_adam             as adam_mod                    # noqa: E402
import solver_cma              as cma_mod                     # noqa: E402
import solver_bayesian         as bay_mod                     # noqa: E402
import solver_hybrid           as hyb_mod                     # noqa: E402

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --------------------------------------------------------------------------
# Cost evaluations charged per recorded step, per method.
#
# The solvers record their cost trace at different granularities, so the trace
# length has to be scaled before the methods can be compared on a single axis.
#
#   GD        17 finite-difference probes + 1 to score the resulting step = 18
#   Adam      18 for the gradient, matching GD
#   CMA       one full population per generation
#   Bayesian  one evaluation per step, by construction
#   Hybrid    one population per generation, as CMA -- but each of those
#             evaluations also solves a linear program, so the count
#             understates the work done. See the note in the module docstring.
# --------------------------------------------------------------------------
EVALS_PER_STEP = {
    'GD':       1 + problem.N_params,
    'Adam':     1 + problem.N_params,
    'CMA':      cma_mod.POPSIZE,
    'Bayesian': 1,
    'Hybrid':   hyb_mod.POPSIZE,
}


def compare(num_trials=1000):
    """Run every method over `num_trials` seeds and produce all comparisons."""
    methods = {
        'GD':       lambda seed: gd_mod.run(seed, show_plot=False)[0],
        'Adam':     lambda seed: adam_mod.run(seed, show_plot=False)[0],
        'CMA':      lambda seed: cma_mod.run(seed, show_plot=False)[0],
        'Bayesian': lambda seed: bay_mod.run(seed, show_plot=False)[0],
        'Hybrid':   lambda seed: hyb_mod.run(seed, show_plot=False)[0],
    }

    results = {name: [] for name in methods}

    # Every method gets the same seeds, so they all face the same set of
    # starting points and the comparison is paired rather than independent.
    all_seeds = list(range(num_trials))

    for name, func in methods.items():
        for trial, seed in enumerate(all_seeds):
            print(f"Seed {seed} | {name} | sim {trial + 1}/{num_trials}")
            t0      = time.time()
            res     = func(seed)
            elapsed = time.time() - t0
            res['wall_time'] = elapsed
            results[name].append(res)
            print(f"  -> {elapsed:.2f}s  |  success={res['success']}  |  "
                  f"final_cost={res['final_cost']:.6f}")

    methods_list = list(methods.keys())

    _plot_success_rate(results, methods_list, num_trials)
    _plot_evaluations(results, methods_list, num_trials)
    _plot_wall_time(results, methods_list, num_trials)
    _print_statistics(results, methods_list, num_trials)

    plt.show()
    return results


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def _scatter_all(ax, i, color, succ_vals, unsucc_vals, all_vals):
    """Scatter successes and failures side by side, with the overall median."""
    if unsucc_vals:
        ax.scatter(np.random.normal(i, 0.04, len(unsucc_vals)), unsucc_vals,
                   alpha=0.4, s=30, color=color, edgecolors='red',
                   linewidth=1, marker='x', zorder=3)
    if succ_vals:
        ax.scatter(np.random.normal(i, 0.04, len(succ_vals)), succ_vals,
                   alpha=0.7, s=35, color=color, edgecolors='black',
                   linewidth=0.5, marker='o', zorder=3)
    if all_vals:
        ax.hlines(np.median(all_vals), i - 0.25, i + 0.25,
                  colors='black', linewidth=2.5, zorder=4)


def _scatter_succ(ax, i, color, succ_vals):
    """Scatter successful runs only, with their median."""
    if succ_vals:
        ax.scatter(np.random.normal(i, 0.04, len(succ_vals)), succ_vals,
                   alpha=0.7, s=35, color=color, edgecolors='black',
                   linewidth=0.5, marker='o', zorder=3)
        ax.hlines(np.median(succ_vals), i - 0.25, i + 0.25,
                  colors='black', linewidth=2.5, zorder=4)


def _format_ax(ax, methods_list, ylabel, title, log=True):
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--')


_LEGEND_BOTH = [
    Line2D([0], [0], marker='o', color='w', label='Successful run',
           markerfacecolor='gray', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='x', color='w', label='Unsuccessful run',
           markerfacecolor='red', markeredgecolor='red', markersize=8),
    Line2D([0], [0], color='black', linewidth=2.5, label='Median (all runs)'),
]
_LEGEND_SUCC = [
    Line2D([0], [0], marker='o', color='w', label='Successful run',
           markerfacecolor='gray', markeredgecolor='black', markersize=8),
    Line2D([0], [0], color='black', linewidth=2.5, label='Median (successful runs)'),
]


def _plot_success_rate(results, methods_list, num_trials):
    """Fraction of runs that reached cost == 0."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sr   = [np.mean([r['success'] for r in results[m]]) * 100 for m in methods_list]
    bars = ax.bar(methods_list, sr, color=COLORS[:len(methods_list)],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Success rate (%)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Success rate (cost = 0)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    for bar, rate in zip(bars, sr):
        h      = bar.get_height()
        n_succ = int(round(rate * num_trials / 100))
        ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2., h / 2,
                f'{n_succ}/{num_trials}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig('success_rate.png', dpi=150, bbox_inches='tight')


def _plot_evaluations(results, methods_list, num_trials):
    """Cost evaluations to converge, split by whether the run succeeded."""
    fig_all,  ax_all  = plt.subplots(figsize=(12, 6))
    fig_succ, ax_succ = plt.subplots(figsize=(12, 6))

    for i, (m, color) in enumerate(zip(methods_list, COLORS)):
        scale      = EVALS_PER_STEP.get(m, 1)
        evals_list = [len(r['cost_arr']) * scale for r in results[m]]
        successes  = [r['success'] for r in results[m]]
        unsucc = [evals_list[j] for j in range(num_trials) if not successes[j]]
        succ   = [evals_list[j] for j in range(num_trials) if     successes[j]]
        _scatter_all(ax_all, i, color, succ, unsucc, evals_list)
        _scatter_succ(ax_succ, i, color, succ)

    _format_ax(ax_all, methods_list, 'Number of cost evaluations',
               'All runs (successful + unsuccessful)', log=True)
    _format_ax(ax_succ, methods_list, 'Number of cost evaluations',
               'Successful runs only', log=True)
    ax_all.legend(handles=_LEGEND_BOTH, loc='upper right', fontsize=10, framealpha=0.9)
    ax_succ.legend(handles=_LEGEND_SUCC, loc='upper right', fontsize=10, framealpha=0.9)
    fig_all.tight_layout()
    fig_all.savefig('evals_to_convergence_all.png', dpi=150, bbox_inches='tight')
    fig_succ.tight_layout()
    fig_succ.savefig('evals_to_convergence_successful.png', dpi=150, bbox_inches='tight')


def _plot_wall_time(results, methods_list, num_trials):
    """Wall-clock time to converge, split by whether the run succeeded."""
    fig_succ, ax_succ = plt.subplots(figsize=(12, 6))
    for i, (m, color) in enumerate(zip(methods_list, COLORS)):
        successes  = [r['success']   for r in results[m]]
        all_times  = [r['wall_time'] for r in results[m]]
        succ_times = [all_times[j] for j in range(num_trials) if successes[j]]
        _scatter_succ(ax_succ, i, color, succ_times)
    _format_ax(ax_succ, methods_list, 'Wall-clock time (seconds)',
               'Wall-clock time - successful runs only', log=True)
    ax_succ.legend(handles=_LEGEND_SUCC, loc='upper right', fontsize=10, framealpha=0.9)
    fig_succ.tight_layout()
    fig_succ.savefig('walltime_successful.png', dpi=150, bbox_inches='tight')

    fig_all, ax_all = plt.subplots(figsize=(12, 6))
    for i, (m, color) in enumerate(zip(methods_list, COLORS)):
        successes  = [r['success']   for r in results[m]]
        all_times  = [r['wall_time'] for r in results[m]]
        succ_times = [all_times[j] for j in range(num_trials) if     successes[j]]
        fail_times = [all_times[j] for j in range(num_trials) if not successes[j]]
        _scatter_all(ax_all, i, color, succ_times, fail_times, all_times)
    _format_ax(ax_all, methods_list, 'Wall-clock time (seconds)',
               'Wall-clock time - all runs (success + failed)', log=True)
    ax_all.legend(handles=_LEGEND_BOTH, loc='upper right', fontsize=10, framealpha=0.9)
    fig_all.tight_layout()
    fig_all.savefig('walltime_all.png', dpi=150, bbox_inches='tight')


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _print_statistics(results, methods_list, num_trials):
    """
    Print medians with their spread and uncertainty.

    Medians rather than means, because a handful of runs that exhaust the
    budget would drag a mean around; the interquartile range describes the
    spread of the runs themselves, while the bootstrap interval describes how
    well the median itself is pinned down.
    """
    rng = np.random.default_rng(42)

    def bootstrap_median_ci(arr, n_boot=10_000, ci=95):
        """Percentile bootstrap confidence interval for a median."""
        if len(arr) == 0:
            return (np.nan, np.nan)
        boots   = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
        medians = np.median(boots, axis=1)
        lo = np.percentile(medians, (100 - ci) / 2)
        hi = np.percentile(medians, 100 - (100 - ci) / 2)
        return (lo, hi)

    def describe(label, arr, fmt):
        if len(arr) == 0:
            return
        lo, hi = bootstrap_median_ci(arr)
        q1, q3 = np.percentile(arr, [25, 75])
        print(f"  {label:15s}: median {np.median(arr):{fmt}}  "
              f"IQR [{q1:{fmt}}, {q3:{fmt}}]  "
              f"95% CI [{lo:{fmt}}, {hi:{fmt}}]  "
              f"mean {np.mean(arr):{fmt}}")

    print("\n" + "=" * 78)
    print("DETAILED STATISTICS")
    print("=" * 78)
    for m in methods_list:
        scale   = EVALS_PER_STEP.get(m, 1)
        n_succ  = sum(r['success'] for r in results[m])
        all_ev  = np.array([len(r['cost_arr']) * scale for r in results[m]])
        succ_ev = np.array([len(r['cost_arr']) * scale for r in results[m] if r['success']])
        all_wt  = np.array([r['wall_time'] for r in results[m]])
        succ_wt = np.array([r['wall_time'] for r in results[m] if r['success']])
        fail_wt = np.array([r['wall_time'] for r in results[m] if not r['success']])

        print(f"\n{m}:")
        print(f"  Success rate   : {100 * n_succ / num_trials:.1f}%  ({n_succ}/{num_trials})")
        describe("Evals all",     all_ev,  ".0f")
        describe("Evals success", succ_ev, ".0f")
        describe("Time all (s)",  all_wt,  ".2f")
        describe("Time succ (s)", succ_wt, ".2f")
        describe("Time fail (s)", fail_wt, ".2f")


if __name__ == "__main__":
    compare(num_trials=1000)
