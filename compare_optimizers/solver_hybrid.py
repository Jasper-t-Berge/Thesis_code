"""
Hybrid CMA-ES + linear-programming solver.

The other four solvers treat all 4*N+M parameters as one opaque search space.
But half of that space does not need to be searched at all.

Every design inequality compares two switching thresholds, and a threshold is

    T(i, s) = h_i(s) / dP(i, s)

where `h_i(s)` is one of valve i's two intrinsic pressures (h+ if the valve is
open in state s, h- if it is closed) and `dP(i, s)` is the fraction of the
source pressure dropped across edge i in state s. The pressure drop is a
property of the *conductances* and the state alone -- see
`calculate_pressure_hysteron_and_conductor_network`, whose only parameters are
C0, dC and the conductors. The threshold parameters never enter it.

So fix the conductances and every threshold becomes a known constant times one
of the remaining unknowns. Each cost term

    max(0, T_a - T_b + epsilon)

is then a hinge on a linear function of those unknowns, the total cost is a
convex piecewise-linear function of them, and its exact minimum is the optimum
of a linear program. That half of the problem stops being a search.

What this file therefore does:

  * CMA-ES searches only conductance space -- 2N+M dimensions instead of 4N+M.
    On the 4-valve benchmark that is 9 instead of 17.
  * For each candidate, an LP returns the smallest cost achievable over *all*
    threshold vectors. That number is the cost handed back to CMA.

The outer search is the same `cma_search` the CMA solver uses, with the same
hyperparameters, so the only thing that changed is what it searches over.

Two notes on fairness of the comparison:

  * One evaluation here is much more expensive than one evaluation elsewhere:
    it solves an LP on top of the same pressure solves. Charged per evaluation
    the hybrid gets a real advantage; the wall-clock comparison is the one that
    prices that in.
  * The LP's feasible box is exactly the region `straighten_parameters` maps
    into, so the hybrid is allowed no thresholds the other solvers are denied.

Everything problem-specific is read from the `problem` module at call time and
cached, so pointing `problem` at a different instance -- a larger network, a
different target -- is enough to retarget the solver. The LP is assembled
sparsely, since the constraint count grows roughly as N^2.
"""

import pathlib
import sys
import time

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import numpy as np                                             # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
import scipy.sparse as sp                                      # noqa: E402
from scipy.optimize import linprog                             # noqa: E402

import problem                                                 # noqa: E402
from hysteron import (                                         # noqa: E402
    HysteronValve,
    calculate_pressure_hysteron_and_conductor_network,
    config,
    convert_number_to_state,
    cost_function_system,
    random_parameter_vector,
    straighten_parameters,
)
from hysteron.optimize import (                                # noqa: E402
    POPSIZE,
    SIGMA0,
    SIGMA_FLOOR,
    STAGNATION_LIMIT,
    cma_search,
)

# --------------------------------------------------------------------------
# The feasible box for the thresholds.
#
# These reproduce exactly what `straighten_parameters` enforces. It works in
# midpoint/size coordinates with minimum midpoint 0.1, minimum size 0.1 and a
# floor of 0.05 on h-, and the image of that map is precisely
#
#     h- >= 0.05   and   h+ - h- >= 0.1
#
# (the midpoint bound follows: 0.05 + 0.1/2 = 0.1). Handing the LP the same
# region is what keeps the comparison honest -- the hybrid may not use a valve
# the other solvers would have had straightened away.
# --------------------------------------------------------------------------
MIN_HM     = 0.05
MIN_H_SIZE = 0.1
H_MAX      = 1e4        # only there to keep the LP bounded; never binding

# A pressure drop this small makes a threshold effectively infinite and the LP
# hopelessly ill-conditioned. It happens on the measure-zero set where a bridge
# is balanced, so the candidate is simply rejected.
P_FLOOR     = 1e-12
REJECT_COST = 1e6

# The margin the LP separates thresholds by.
#
# `config.EPSILON` is 1e-9, which is deliberately far below the scale of the
# thresholds -- but it is also below HiGHS's own primal feasibility tolerance
# of about 1e-7. Asking the LP to respect a margin finer than the tolerance it
# solves to means it can report a constraint as satisfied when it is not, and
# the run would then be scored a success on a network that does not quite work.
#
# So the LP is given a coarser margin than the cost function demands. That is
# safe in the direction that matters: a threshold ordering separated by 1e-5
# is separated by 1e-9 too, so an LP optimum of zero is a genuine zero of the
# real cost, never a false one. Nothing is given up either, because the cost is
# homogeneous in the thresholds -- any strict ordering can be scaled up until
# it clears the coarser margin, and `H_MAX` leaves four decades of room to do
# it in. 1e-5 still sits four orders below the 0.1-10 scale of the thresholds
# being separated, so it is no more of a physical margin than 1e-9 was.
#
# Taking the max keeps the LP from ever being the *looser* of the two, should
# `config.EPSILON` be raised to something that really is a physical margin.
LP_EPSILON = max(1e-5, config.EPSILON)


# --------------------------------------------------------------------------
# Precomputed LP structure
# --------------------------------------------------------------------------

class _Structure:
    """
    Everything about the LP that the conductances do not change.

    Which variable and which pressure drop each inequality refers to is fixed
    by the target t-graph, so it is worked out once per problem instance. Only
    the coefficients change from one candidate to the next.
    """

    def __init__(self, prob):
        self.prob   = prob
        self.ineq   = prob.ineq_array
        self.NV     = prob.N_valves
        self.N_COND = 2 * self.NV + prob.N_conds
        self.N_THR  = 2 * self.NV
        self.K      = len(self.ineq)

        NV, K = self.NV, self.K
        self.states = np.unique(self.ineq[:, :, 2])
        row_of      = {int(s): i for i, s in enumerate(self.states)}
        self.bits   = np.array([convert_number_to_state(int(s), NV)
                                for s in self.states])

        def side(k):
            """
            Resolve one side of every inequality into (variable, state, valve).

            A valve that is open in the referenced state can only close, so h+
            is the threshold that applies; a closed one can only re-open, so h-
            applies. That is the same rule
            `calculate_hysteron_thresholds_given_state` follows, and it is why
            the inequalities never have to record which threshold they mean.
            """
            valve  = self.ineq[:, k, 1].astype(int)
            row    = np.array([row_of[int(s)] for s in self.ineq[:, k, 2]])
            closed = self.bits[row, valve] == 1
            return np.where(closed, valve, NV + valve), row, valve

        self.var_s, self.row_s, self.valve_s = side(0)   # must be the smaller
        self.var_l, self.row_l, self.valve_l = side(1)   # must be the larger

        # LP variables: [ thresholds (2N) | slacks, one per inequality ].
        self.n_var = self.N_THR + K
        self.obj   = np.concatenate([np.zeros(self.N_THR), np.ones(K)])

        # Right-hand side: the separation margin, then the minimum hysteresis
        # width h+ - h- >= 0.1 for each valve.
        self.b = np.concatenate([np.full(K, -LP_EPSILON),
                                 np.full(NV, -MIN_H_SIZE)])

        # h- >= 0.05, h+ >= 0.15 (it must clear h- by the minimum width),
        # slacks >= 0.
        self.bounds = ([(MIN_HM, H_MAX)] * NV
                       + [(MIN_HM + MIN_H_SIZE, H_MAX)] * NV
                       + [(0.0, None)] * K)

        # The constraint matrix is assembled sparsely: it has O(N^2) rows but
        # only three nonzeros per row, so dense storage would be quadratic
        # waste. The part that does not depend on the candidate -- the -1 on
        # each slack and the two entries of each width constraint -- is stored
        # once as triplets and reused.
        k_ix = np.arange(K)
        i_ix = np.arange(NV)
        self.fixed_r = np.concatenate([k_ix, K + i_ix, K + i_ix])
        self.fixed_c = np.concatenate([self.N_THR + k_ix, i_ix, NV + i_ix])
        self.fixed_v = np.concatenate([-np.ones(K), np.ones(NV), -np.ones(NV)])
        self.k_ix    = k_ix
        self.shape   = (K + NV, self.n_var)


_struct = None


def _structure():
    """The cached LP structure for whatever `problem` currently describes."""
    global _struct
    if _struct is None or _struct.ineq is not problem.ineq_array:
        _struct = _Structure(problem)
    return _struct


# --------------------------------------------------------------------------
# Conductance side
# --------------------------------------------------------------------------

def straighten_conductances(q):
    """
    Project a raw conductance vector onto the physical region.

    `straighten_parameters` treats the threshold and conductance halves of the
    parameter vector completely independently, so it is reused here on a vector
    whose threshold half is a placeholder. Reusing it rather than reimplementing
    the projection is deliberate: the hybrid then lives under exactly the same
    constraints as every other solver, by construction rather than by matching
    two copies of the same arithmetic.
    """
    st = _structure()
    padded = np.concatenate([np.zeros(st.N_THR), np.asarray(q, dtype=float)])
    valves = HysteronValve(np.zeros(st.NV), None, None, None, None)
    padded, valves, _ = straighten_parameters(padded, valves, None)
    return padded[st.N_THR:]


def pressure_table(q_straight):
    """
    Pressure drop across every valve, in every state the inequalities mention.

    Returns an (n_states, N_valves) array. This is the whole of the candidate's
    physics: everything the LP needs to know about the conductances is in here.
    """
    st = _structure()
    NV = st.NV
    valves = HysteronValve(np.zeros(NV), None, None,
                           q_straight[:NV], q_straight[NV:2 * NV])
    conds = q_straight[2 * NV:]

    table = np.zeros((len(st.states), NV))
    for i in range(len(st.states)):
        valves.state = st.bits[i]
        table[i] = calculate_pressure_hysteron_and_conductor_network(
            valves, conds, problem.boundary_pressure_template,
            problem.incidence_matrix)[:NV]
    return table


# --------------------------------------------------------------------------
# Threshold side: the linear program
# --------------------------------------------------------------------------

def _solve_lp(q_straight):
    """
    The linear program itself, on conductances that are already straightened.

    Split out from `solve_thresholds` only so the callers that need the
    straightened conductances anyway do not have to project them twice; this is
    the inner loop of the whole method.

    The LP is

        minimise    sum_k  slack_k
        subject to  c_small,k * h_small,k - c_large,k * h_large,k
                     + LP_EPSILON  <=  slack_k          for every inequality k
                    h+_i - h-_i >= 0.1                  for every valve i
                    slack_k >= 0,  h-_i >= 0.05

    which is the standard epigraph form of the piecewise-linear cost: at the
    optimum each slack sits exactly at `max(0, ...)`, since nothing pushes it
    higher and the two constraints stop it going lower. The optimum is
    therefore the true minimum of the cost over all threshold vectors, not an
    approximation to it -- up to the margin, which is `LP_EPSILON` rather than
    `config.EPSILON` for the reason given where it is defined. The returned
    cost is consequently a slight over-estimate on candidates that do not
    reach zero, and exact on the ones that do.
    """
    st    = _structure()
    table = pressure_table(q_straight)

    if not np.all(np.isfinite(table)) or np.min(np.abs(table)) < P_FLOOR:
        return None, REJECT_COST

    coeff_small = 1.0 / table[st.row_s, st.valve_s]
    coeff_large = 1.0 / table[st.row_l, st.valve_l]

    rows = np.concatenate([st.fixed_r, st.k_ix, st.k_ix])
    cols = np.concatenate([st.fixed_c, st.var_s, st.var_l])
    vals = np.concatenate([st.fixed_v, coeff_small, -coeff_large])
    A = sp.csr_matrix((vals, (rows, cols)), shape=st.shape)

    res = linprog(st.obj, A_ub=A, b_ub=st.b, bounds=st.bounds, method='highs')

    if not res.success:
        return None, REJECT_COST

    return res.x[:st.N_THR], float(res.fun)


def solve_thresholds(q):
    """
    Best thresholds for a raw conductance vector, and the cost they achieve.

    Returns `(h, cost)` with `h` the optimal threshold vector laid out as
    [h- x N | h+ x N], or `(None, REJECT_COST)` if the candidate is degenerate.
    """
    return _solve_lp(straighten_conductances(q))


def full_params(q):
    """
    Assemble the complete parameter vector a candidate stands for.

    The conductances are straightened and the thresholds come from the LP, so
    the result is already a fixed point of `straighten_parameters` and can be
    used as-is.
    """
    st = _structure()
    q_straight = straighten_conductances(q)
    h, _ = _solve_lp(q_straight)
    if h is None:
        # Degenerate candidate: no thresholds were solved for, so fill in a
        # placeholder that at least satisfies the physical constraints.
        h = np.concatenate([np.full(st.NV, MIN_HM),
                            np.full(st.NV, MIN_HM + MIN_H_SIZE)])
    return np.concatenate([h, q_straight])


def reconstruct_valves_conds(p):
    """
    Build a `HysteronValve` and conductor array from a full parameter vector.

    Mirrors the CMA solver's function of the same name so that a hybrid result
    can be fed to the same downstream code.
    """
    NV = _structure().NV
    p = np.array(p, dtype=float)
    v = HysteronValve(np.zeros(NV), p[:NV], p[NV:2 * NV],
                      p[2 * NV:3 * NV], p[3 * NV:4 * NV])
    c = p[4 * NV:].copy()
    p, v, c = straighten_parameters(p, v, c)
    return v, c


def hybrid_cost(q):
    """
    Cost of one candidate conductance vector.

    The LP's own objective is the answer, but the cost actually reported is
    recomputed with `cost_function_system` on the reconstructed network. That
    keeps the number on exactly the same footing as every other solver's: the
    hybrid is scored by the shared cost function, not by its own internal
    estimate of it.
    """
    st = _structure()
    NV = st.NV
    q_straight = straighten_conductances(q)
    h, _ = _solve_lp(q_straight)
    if h is None:
        return REJECT_COST

    valves = HysteronValve(np.zeros(NV), h[:NV], h[NV:],
                           q_straight[:NV], q_straight[NV:2 * NV])
    return cost_function_system(problem.ineq_array, valves, q_straight[2 * NV:],
                                problem.boundary_pressure_template,
                                problem.incidence_matrix)


def fresh_x0():
    """
    Draw a random starting point in conductance space.

    The draw goes through `random_parameter_vector` and then discards the
    threshold half, rather than drawing the conductances directly. That keeps
    the random stream identical to the CMA solver's, so a given seed hands both
    methods the same conductances to start from and the comparison between them
    is paired.
    """
    st = _structure()
    params, _, _ = random_parameter_vector(
        st.NV, problem.N_conds,
        problem.H_LOWER_BOUND, problem.H_UPPER_BOUND,
        problem.C_LOWER_BOUND, problem.C_UPPER_BOUND)
    return params[st.N_THR:]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(seed, max_fevals=None, num_sim=1, sigma0=SIGMA0,
        sigma_floor=SIGMA_FLOOR, stagnation_limit=STAGNATION_LIMIT,
        data_filename=None, show_plot=True):
    """
    Run the hybrid solver.

    Same signature and same return format as `solver_cma.run`, so the two are
    interchangeable in `run_comparison`. `params` is the full parameter vector,
    with the LP's thresholds already substituted in.
    """
    if max_fevals is None:
        max_fevals = problem.EVAL_BUDGET
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
                    print(f"  Hybrid sim {_j}: {100 * milestone:.0f}% done, "
                          f"cost = {best_cost_overall:.6f}")

        res = cma_search(hybrid_cost, fresh_x0, max_fevals,
                         sigma0=sigma0, sigma_floor=sigma_floor,
                         stagnation_limit=stagnation_limit,
                         on_generation=report)

        cost_arr = res['cost_arr']
        if res['completion_index'] != -1:
            print(f"  Hybrid sim {j}: solution found at generation "
                  f"{res['completion_index']}!")
        print(f"  Hybrid sim {j}: completion={res['completion_index']}, "
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
            'params':           full_params(res['best_params']),
        })

    if show_plot:
        plt.figure()
        plt.title("Hybrid CMA-ES + LP - best cost per generation")
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
