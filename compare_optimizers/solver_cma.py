"""
CMA-ES solver (Covariance Matrix Adaptation Evolution Strategy).

The only gradient-free method of the four. Instead of probing the cost around
a single point, it samples a whole population from a multivariate Gaussian,
keeps the better half, and moves and reshapes that Gaussian towards them. Over
successive generations the distribution stretches along whichever directions
keep producing improvements, which suits a landscape whose parameters are
strongly coupled and which has no usable gradient.

The search itself lives in `hysteron.optimize`, shared with the topology
sweep so that both run the same optimiser. This file supplies the pieces
specific to the benchmark: the cost function, the starting points, and how a
raw candidate becomes a physical network.

Hyperparameters: population 12, of which the best 6 update the distribution;
initial step size 0.3 with a floor of 0.01; restart after 100 generations
without improvement.
"""

import pathlib
import sys
import time

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import numpy as np                                            # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

import problem                                                # noqa: E402
from hysteron import (                                        # noqa: E402
    HysteronValve,
    cost_function_system,
    random_parameter_vector,
    straighten_parameters,
)
from hysteron.optimize import (                               # noqa: E402
    MU,
    POPSIZE,
    SIGMA0,
    SIGMA_FLOOR,
    STAGNATION_LIMIT,
    cma_search,
)


def reconstruct_valves_conds(p):
    """
    Build a `HysteronValve` and conductor array from a raw CMA candidate.

    CMA-ES samples an unconstrained Gaussian, so candidates routinely violate
    the physical constraints; `straighten_parameters` projects them back.
    """
    p = np.array(p, dtype=float)
    v = HysteronValve(
        np.zeros(problem.N_valves),
        p[:problem.N_valves],
        p[problem.N_valves:2 * problem.N_valves],
        p[2 * problem.N_valves:3 * problem.N_valves],
        p[3 * problem.N_valves:4 * problem.N_valves],
    )
    c = p[4 * problem.N_valves:].copy()
    p, v, c = straighten_parameters(p, v, c)
    return v, c


def cma_cost(p):
    """Cost of a single CMA candidate."""
    v, c = reconstruct_valves_conds(p)
    return cost_function_system(problem.ineq_array, v, c,
                                problem.boundary_pressure_template,
                                problem.incidence_matrix)


def fresh_x0():
    """Draw a random starting point in the physical parameter space."""
    params, _, _ = random_parameter_vector(
        problem.N_valves, problem.N_conds,
        problem.H_LOWER_BOUND, problem.H_UPPER_BOUND,
        problem.C_LOWER_BOUND, problem.C_UPPER_BOUND)
    return params


def run(seed, max_fevals=problem.EVAL_BUDGET, num_sim=1, sigma0=SIGMA0,
        sigma_floor=SIGMA_FLOOR, stagnation_limit=STAGNATION_LIMIT,
        data_filename=None, show_plot=True):
    """
    Run the CMA-ES solver.

    Unlike the step-based solvers, the budget here is counted directly in cost
    evaluations, since a generation costs exactly `POPSIZE` of them.

    Returns a list of result dicts, one per simulation, with keys:
        success, final_cost, completion_index, num_restarts, cost_arr, params

    `params` is the best parameter vector found, which is the actual answer
    to the design problem; the rest describe how the search got there. It is
    a raw candidate, so pass it through `reconstruct_valves_conds` before use.
    """
    # numpy rejects a negative seed, so translate the -1 convention used by
    # the other solvers into a clock-derived seed here.
    if seed == -1:
        seed = int(time.time() * 10000) % 100000000
        print('seed:', seed, "\n")
    np.random.seed(seed)
    all_results = []

    for j in range(num_sim):

        def report(fevals, n_new, best_cost_overall, _j=j):
            """Print a line at each quarter of the budget."""
            progress      = fevals / max_fevals
            prev_progress = (fevals - n_new) / max_fevals
            for milestone in (0.25, 0.50, 0.75):
                if prev_progress < milestone <= progress:
                    print(f"  CMA sim {_j}: {100 * milestone:.0f}% done, "
                          f"cost = {best_cost_overall:.6f}")

        res = cma_search(cma_cost, fresh_x0, max_fevals,
                         sigma0=sigma0, sigma_floor=sigma_floor,
                         stagnation_limit=stagnation_limit,
                         on_generation=report)

        cost_arr = res['cost_arr']
        if res['completion_index'] != -1:
            print(f"  CMA sim {j}: solution found at generation "
                  f"{res['completion_index']}!")
        print(f"  CMA sim {j}: completion={res['completion_index']}, "
              f"restarts={res['num_restarts']}, final cost={cost_arr[-1]:.6f}")

        if data_filename:
            with open(data_filename, "a") as f:
                f.write(f"{cost_arr[0]},{cost_arr[-1]},"
                        f"{res['completion_index']},{res['num_restarts']}\n")

        all_results.append({
            'success':          res['success'],
            'final_cost':       res['best_cost'],
            'completion_index': res['completion_index'],
            'num_restarts':     res['num_restarts'],
            'cost_arr':         cost_arr,
            'params':           res['best_params'],
        })

    if show_plot:
        plt.figure()
        plt.title("CMA-ES - best cost per generation")
        plt.xlabel("Generation")
        plt.ylabel("Best cost so far")
        plt.plot(cost_arr)
        plt.yscale('log')
        plt.tight_layout()
        plt.show(block=True)

    return all_results


if __name__ == "__main__":
    results = run(seed=-1, num_sim=1, show_plot=True)
    r = results[0]
    print(f"Success:           {r['success']}")
    print(f"Final cost:        {r['final_cost']:.6f}")
    print(f"Completion index:  {r['completion_index']}")
    print(f"Restarts:          {r['num_restarts']}")
