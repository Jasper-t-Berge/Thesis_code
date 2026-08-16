"""
A scalable hard instance: a Wheatstone ladder driven by partial pressure pulses.

Two ingredients make it demanding rather than Preisach-easy:

  * Topology. The 4-valve MPO needed the Wheatstone bridge, because only its
    coupling is rich enough. This chains k Wheatstone cells in series, so the
    coupling stays rich inside each cell while the cells throttle each other.

  * Protocol. A full major loop only ever visits ~2N states and constrains a
    valve in one or two of them, which is why random conductances solve it.
    Partial pulses -- up to a fraction of saturation, back down to a nonzero
    floor -- revisit states and pin the *same* valve's two thresholds against
    many different pressure drops, which is what over-constrains the problem.

The target is read off a randomly drawn network, so it is realisable by
construction: the generating network scores exactly zero on it.

A `ScaledProblem` exposes the same attribute names as `problem`, so pointing
the solvers at one is a matter of copying its attributes onto that module;
`run_scaling.py` does exactly that.
"""
import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron`
# resolves however this file is launched. Matches `problem.py`.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import numpy as np                                             # noqa: E402

from hysteron import (                                         # noqa: E402
    HysteronValve, build_incidence_matrix, build_inequalities,
    calculate_pressure_hysteron_and_conductor_network, config,
    convert_number_to_state, convert_state_to_number, cost_function_system,
    find_threshold_down_transition, find_threshold_up_transition,
    random_parameter_vector, straighten_parameters)
from hysteron.tgraph import check_for_avalanches                # noqa: E402

H_LOWER_BOUND, H_UPPER_BOUND = 0.1, 1.0
C_LOWER_BOUND, C_UPPER_BOUND = 0.5, 5.0


def wheatstone_ladder(k):
    """
    k Wheatstone cells in series: 4k valves, k conductors.

    Cell j runs from its input node to its output node through two internal
    nodes A and B, wired exactly like the 4-valve bridge -- I->A, A->B (the
    bridge), B->O, A->O, plus the conductor I->B. Cell outputs feed the next
    cell's input, so the whole ladder is a chain of bridges.
    """
    N_valves, N_conds = 4 * k, k
    A = [2 + 2 * j for j in range(k)]           # internal node A of cell j
    B = [3 + 2 * j for j in range(k)]           # internal node B of cell j
    J = [2 + 2 * k + j for j in range(k - 1)]   # junctions between cells
    N_nodes = 2 + 2 * k + (k - 1)

    def I_of(j): return 1 if j == 0 else J[j - 1]
    def O_of(j): return 0 if j == k - 1 else J[j]

    edges = np.zeros((N_valves + N_conds, 2), dtype=int)
    for j in range(k):
        edges[4 * j + 0] = (I_of(j), A[j])
        edges[4 * j + 1] = (A[j],    B[j])      # the bridge
        edges[4 * j + 2] = (B[j],    O_of(j))
        edges[4 * j + 3] = (A[j],    O_of(j))
        edges[N_valves + j] = (I_of(j), B[j])
    return edges, N_valves, N_conds, N_nodes


def _step_up(valves, conds, bpt, inc, s, ceiling):
    """One up-transition from state s, or None if nothing flips below ceiling."""
    N = len(valves.state)
    valves.state = convert_number_to_state(s, N)
    pe = calculate_pressure_hysteron_and_conductor_network(valves, conds, bpt, inc)
    thr, idx = find_threshold_up_transition(valves, pe, index_flag=True)
    if idx == -1 or thr > ceiling or not np.isfinite(thr):
        return None
    valves.state[idx] = (valves.state[idx] + 1) % 2
    valves = check_for_avalanches(valves, conds, bpt, inc, thr)
    nxt = convert_state_to_number(valves.state)
    return None if nxt == s else (nxt, thr)


def _step_down(valves, conds, bpt, inc, s, floor):
    """One down-transition from state s, or None if nothing flips above floor."""
    N = len(valves.state)
    valves.state = convert_number_to_state(s, N)
    pe = calculate_pressure_hysteron_and_conductor_network(valves, conds, bpt, inc)
    thr, idx = find_threshold_down_transition(valves, pe, index_flag=True)
    if idx == -1 or thr < floor or not np.isfinite(thr):
        return None
    valves.state[idx] = (valves.state[idx] + 1) % 2
    valves = check_for_avalanches(valves, conds, bpt, inc, thr)
    nxt = convert_state_to_number(valves.state)
    return None if nxt == s else (nxt, thr)


def saturation_threshold(valves, conds, bpt, inc):
    """Highest source pressure any transition needs, from a full up-sweep."""
    N = len(valves.state)
    s, top = 0, 0.0
    for _ in range(4 * N):
        r = _step_up(valves, conds, bpt, inc, s, np.inf)
        if r is None:
            break
        s, thr = r[0], r[1]
        top = max(top, thr)
    return top if top > 0 else 1.0


def pulse_target(valves, conds, bpt, inc, n_pulses, rng):
    """
    Drive the network with partial pulses and record every transition seen.

    Each pulse rises to a random fraction of saturation and falls back to a
    random nonzero floor, so the sweep keeps turning around part-way and
    revisiting states rather than running to the rails.
    """
    T = saturation_threshold(valves, conds, bpt, inc)
    N = len(valves.state)
    up, down, s = {}, {}, 0

    for _ in range(n_pulses):
        ceiling = T * rng.uniform(0.35, 1.05)
        floor   = T * rng.uniform(0.0, 0.45)
        for _ in range(4 * N):
            r = _step_up(valves, conds, bpt, inc, s, ceiling)
            if r is None:
                break
            up[s] = r[0]; s = r[0]
        for _ in range(4 * N):
            r = _step_down(valves, conds, bpt, inc, s, floor)
            if r is None:
                break
            down[s] = r[0]; s = r[0]

    # Keep only single-flip transitions.
    #
    # `inequalities_up_transition` handles a cascade by looping over every
    # flipped bit as the critical one, so it demands each of them be the first
    # to flip out of n1. Physically only one valve triggers and the others
    # follow, so a network that genuinely produces the avalanche still scores
    # a nonzero cost on its own behaviour -- the encoding is stricter than the
    # physics. Single-flip transitions are the regime it is exact on, and they
    # are what the 4-valve MPO target consists of, so restricting to them keeps
    # this instance faithful to the benchmark it is scaling up.
    single = lambda a, b: bin(a ^ b).count('1') == 1
    return ([[a, b] for a, b in up.items()   if single(a, b)],
            [[a, b] for a, b in down.items() if single(a, b)])


class ScaledProblem:
    """
    Drop-in replacement for the `problem` module, at any ladder size.

    Exposes the same attribute names the solvers read, so they can be pointed
    at this instance by monkeypatching and otherwise run unmodified.
    """

    # Targets are generated from seeds in a range disjoint from the trial
    # seeds. `random_parameter_vector` is seeded the same way here as in every
    # solver's `fresh_x0`, so a target seed that collided with a trial seed
    # would hand that trial the exact network the target was read off -- a
    # guaranteed free success. TARGET_SEED_BASE keeps the two streams apart.
    TARGET_SEED_BASE = 100_000

    def __init__(self, k, seed=0, n_pulses=6, eval_budget=10_000):
        self.k = k
        edges, NV, NC, NN = wheatstone_ladder(k)
        self.N_valves, self.N_conds = NV, NC
        self.N_edges  = NV + NC
        self.N_params = 4 * NV + NC
        self.incidence_matrix = build_incidence_matrix(edges, NV + NC, NN)
        self.boundary_pressure_template = config.boundary_pressure_template
        self.EVAL_BUDGET = eval_budget
        self.H_LOWER_BOUND, self.H_UPPER_BOUND = H_LOWER_BOUND, H_UPPER_BOUND
        self.C_LOWER_BOUND, self.C_UPPER_BOUND = C_LOWER_BOUND, C_UPPER_BOUND

        target_seed = self.TARGET_SEED_BASE + seed
        rng = np.random.default_rng(target_seed)
        np.random.seed(target_seed)
        p, v, c = random_parameter_vector(NV, NC, H_LOWER_BOUND, H_UPPER_BOUND,
                                          C_LOWER_BOUND, C_UPPER_BOUND)
        p, v, c = straighten_parameters(p, v, c)
        self.gen_params = p.copy()

        up, down = pulse_target(v, c, self.boundary_pressure_template,
                                self.incidence_matrix, n_pulses, rng)
        self.up_trans, self.down_trans = up, down
        self.ineq_array = build_inequalities(NV, up, down)
        self.n_states = len(np.unique(self.ineq_array[:, :, 2]))

        gv = HysteronValve(np.zeros(NV), p[:NV], p[NV:2*NV],
                           p[2*NV:3*NV], p[3*NV:4*NV])
        self.gen_cost = cost_function_system(
            self.ineq_array, gv, p[4*NV:], self.boundary_pressure_template,
            self.incidence_matrix)
