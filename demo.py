"""
A short demonstration of the whole pipeline, start to finish.

Solves one inverse design problem, then checks and draws the answer:

  1. pick a topology and a target behaviour
  2. run CMA-ES until the cost reaches zero
  3. print the valve parameters it found
  4. sweep the pressure over the solved network to read off the t-graph it
     actually produces, and confirm every targeted transition is there
  5. draw that t-graph

Takes a few seconds. Run it to see what the rest of the project does:

    python demo.py
"""

import pathlib
import sys

# Put the project root and the solver folder on the import path, so this runs
# from anywhere without the project needing to be installed.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE / "compare_optimizers")]

import matplotlib.pyplot as plt                               # noqa: E402
import numpy as np                                            # noqa: E402

import problem                                                # noqa: E402
import solver_cma                                             # noqa: E402
from hysteron import find_t_graph_with_thresholds             # noqa: E402
from hysteron.plotting import plot_t_graph, print_all_hysterons  # noqa: E402

# A seed that converges quickly, so the demo stays short. Any seed works;
# most solve within a few hundred evaluations.
SEED = 3


def main():
    print("=" * 70)
    print(f"Target: {problem.TARGET_TGRAPH} behaviour on the "
          f"{problem.TOPOLOGY} topology")
    print(f"Searching {problem.N_params} parameters against "
          f"{len(problem.ineq_array)} design inequalities")
    print("=" * 70)

    # --- 1. Solve -------------------------------------------------------
    result = solver_cma.run(seed=SEED, num_sim=1, show_plot=False)[0]

    if not result['success']:
        print(f"\nNo solution within the budget (final cost "
              f"{result['final_cost']:.6g}). Try a different SEED.")
        return

    generations = result['completion_index']
    print(f"\nSolved after {generations} generations "
          f"({generations * solver_cma.POPSIZE} cost evaluations).")

    # --- 2. Inspect the solved network ----------------------------------
    # `params` is the answer to the design problem. CMA hands back a raw
    # candidate, so straighten it into physical valves and conductances.
    valves, conds = solver_cma.reconstruct_valves_conds(result['params'])

    print("\nValve parameters found:")
    print_all_hysterons(valves)
    print(f"conductor conductance: {np.round(conds, 6)}")

    # --- 3. Read the behaviour back off the network ---------------------
    # This is the forward problem, and it is the real check: the optimiser
    # minimised a proxy (the inequalities), so sweeping the pressure over the
    # solved network is what confirms the intended behaviour is really there.
    t_graph = find_t_graph_with_thresholds(
        valves, conds, problem.boundary_pressure_template,
        problem.incidence_matrix)

    up_map   = {int(r[0]): int(r[2]) for r in t_graph}
    down_map = {int(r[0]): int(r[1]) for r in t_graph}

    print(f"\nThe solved network reaches {len(t_graph)} stable states.")
    print("\nChecking every targeted transition against what it actually does:")

    all_ok = True
    for label, targets, realised in (("up  ", problem.up_trans,   up_map),
                                     ("down", problem.down_trans, down_map)):
        for src, dst in targets:
            got = realised.get(src)
            ok  = (got == dst)
            all_ok &= ok
            print(f"  {label} {src:>2} -> {dst:<2}   "
                  f"{'ok' if ok else f'MISMATCH, got {got}'}")

    print("\nEvery targeted transition reproduced."
          if all_ok else
          "\nSome transitions differ from the target.")

    # --- 4. Draw it -----------------------------------------------------
    # Blue edges are up-transitions, red are down. Node labels show the
    # open/closed pattern of the four valves.
    plot_t_graph(t_graph, blockbool=False, spinshow=True)
    plt.savefig('demo_tgraph.png', dpi=150, bbox_inches='tight')
    print("\nt-graph written to demo_tgraph.png")
    plt.show()


if __name__ == "__main__":
    main()
