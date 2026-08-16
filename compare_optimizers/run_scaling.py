"""
How the methods hold up as the network grows.

`run_comparison.py` fixes the problem at 4 valves and asks which optimiser is
best. This asks the other question: what happens to each of them as the system
gets larger. It sweeps a Wheatstone ladder from 4 up to 60 valves and runs the
three methods that are fast enough to be worth the wall-clock -- Adam, CMA and
the hybrid.

The instance at each size comes from `scaled_problem.ScaledProblem`, which
reads its target off a randomly drawn network so that a solution is guaranteed
to exist. The solvers are pointed at it by copying its attributes onto the
`problem` module, which is why they need no modification to run at any size:
every one of them reads `problem` at call time.

Two things are worth knowing before reading the output.

  * The evaluation budget is held at 10,000 for every size and every method, so
    the columns stay comparable. Adam's step count is derived from it, since a
    step costs `1 + N_params` evaluations and `N_params` grows with the network.
  * A hybrid evaluation costs far more than the others, because it solves a
    linear program. The evaluation columns therefore flatter it and the
    wall-clock columns are the ones that price it honestly.

Every claimed success is re-verified: the network is rebuilt from the returned
parameter vector and re-scored with `cost_function_system`. That check is not
ceremonial -- it is what catches a solver reporting a cost it did not actually
achieve with the parameters it handed back.

Usage:

    python compare_optimizers/run_scaling.py [cells] [seeds] [pulses] [budget]

`cells` is a comma-separated list of ladder sizes k, each giving 4k valves.
"""

import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import contextlib                                              # noqa: E402
import io                                                      # noqa: E402
import statistics as stats                                     # noqa: E402
import time                                                    # noqa: E402

import numpy as np                                             # noqa: E402

import problem                                                 # noqa: E402
import solver_adam   as adam_mod                               # noqa: E402
import solver_cma    as cma_mod                                # noqa: E402
import solver_hybrid as hyb_mod                                # noqa: E402
from hysteron import cost_function_system                      # noqa: E402
from scaled_problem import ScaledProblem                       # noqa: E402

# The attributes a solver reads off the `problem` module. Retargeting is just
# copying these across from a `ScaledProblem`.
PROBLEM_ATTRS = (
    'N_valves', 'N_conds', 'N_params', 'ineq_array', 'incidence_matrix',
    'boundary_pressure_template', 'EVAL_BUDGET', 'H_LOWER_BOUND',
    'H_UPPER_BOUND', 'C_LOWER_BOUND', 'C_UPPER_BOUND',
)


def run_scaling(cells=(1, 2, 4, 8), n_seeds=10, n_pulses=40, budget=10_000):
    """Run Adam, CMA and the hybrid at each ladder size, printing as it goes."""
    print(f"scaling sweep | cells={list(cells)} seeds={n_seeds} "
          f"pulses={n_pulses} budget={budget}")
    print("=" * 96, flush=True)
    print(f"{'k':>3}{'N':>4}{'dimCMA':>8}{'dimHyb':>8}{'ineq':>7}  "
          f"{'method':<8}{'success':>9}{'ev(all)':>9}{'ev(suc)':>9}"
          f"{'wall(all)':>11}{'wall(suc)':>11}{'ver':>7}{'took':>8}")
    print("-" * 96, flush=True)

    results = {}

    for k in cells:
        inst = ScaledProblem(k, seed=0, n_pulses=n_pulses, eval_budget=budget)
        for attr in PROBLEM_ATTRS:
            setattr(problem, attr, getattr(inst, attr))

        # The target was read off a real network, so a zero-cost solution
        # exists by construction. If this trips, the instance is malformed and
        # nothing below it means anything.
        assert inst.gen_cost == 0, f"target not realisable at k={k}: {inst.gen_cost}"

        NV, NP = inst.N_valves, inst.N_params
        adam_steps = max(1, budget // (1 + NP))
        runners = {
            'Adam':   lambda s: adam_mod.run(s, num_steps=adam_steps,
                                             show_plot=False)[0],
            'CMA':    lambda s: cma_mod.run(s, max_fevals=budget,
                                            show_plot=False)[0],
            'Hybrid': lambda s: hyb_mod.run(s, max_fevals=budget,
                                            show_plot=False)[0],
        }
        # Evaluations charged per recorded step, as in `run_comparison`.
        per_step = {'Adam': 1 + NP, 'CMA': cma_mod.POPSIZE,
                    'Hybrid': hyb_mod.POPSIZE}

        first = True
        for name, runner in runners.items():
            t_method = time.time()
            runs = []
            for seed in range(n_seeds):
                t_run = time.time()
                # The solvers print their own progress; swallow it so the
                # table stays readable.
                with contextlib.redirect_stdout(io.StringIO()):
                    res = runner(seed)
                res['wall_time'] = time.time() - t_run
                runs.append(res)

            n_succ  = sum(r['success'] for r in runs)
            ev_all  = [len(r['cost_arr']) * per_step[name] for r in runs]
            ev_succ = [e for e, r in zip(ev_all, runs) if r['success']]
            wt_all  = [r['wall_time'] for r in runs]
            wt_succ = [r['wall_time'] for r in runs if r['success']]

            verified = 0
            for r in runs:
                if not r['success']:
                    continue
                valves, conds = hyb_mod.reconstruct_valves_conds(
                    np.asarray(r['params'], dtype=float))
                verified += (cost_function_system(
                    inst.ineq_array, valves, conds,
                    inst.boundary_pressure_template,
                    inst.incidence_matrix) == 0)

            results[(k, name)] = {
                'n_succ': n_succ, 'verified': verified, 'runs': runs,
                'ev_all': ev_all, 'ev_succ': ev_succ,
            }

            med = lambda a, f="{:.0f}": f.format(stats.median(a)) if a else "-"
            lead = (f"{k:>3}{NV:>4}{NP:>8}{2 * NV + inst.N_conds:>8}"
                    f"{len(inst.ineq_array):>7}  " if first else " " * 32)
            first = False
            print(f"{lead}{name:<8}{f'{n_succ}/{n_seeds}':>9}"
                  f"{med(ev_all):>9}{med(ev_succ):>9}"
                  f"{med(wt_all, '{:.2f}') + 's':>11}"
                  f"{med(wt_succ, '{:.2f}') + 's':>11}"
                  f"{f'{verified}/{n_succ}':>7}"
                  f"{time.time() - t_method:>7.0f}s", flush=True)
        print("-" * 96, flush=True)

    return results


if __name__ == "__main__":
    cells    = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 2, 4, 8]
    n_seeds  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_pulses = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    budget   = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
    run_scaling(cells, n_seeds, n_pulses, budget)
