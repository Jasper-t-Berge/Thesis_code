"""
The parameter vector: how it is laid out, initialised, and kept physical.

Every optimiser in this project searches the same flat vector of length
4 * N_valves + N_conds (17 for all three topologies studied). The layout is
fixed everywhere:

    [ h-  x N_valves | h+  x N_valves | C0  x N_valves | dC  x N_valves | conds ]

`straighten_parameters` is what keeps that vector physically meaningful, and
`give_kick` is the stagnation escape used by the gradient descent, Adam and
TPE solvers.
"""

import numpy as np

from .network import HysteronValve


def initialize_random_hysteron_valves(N_valves, h_lower_bound, h_upper_bound,
                                      C_lower, C_upper):
    """
    Draw a random starting point for one bank of valves.

    Two independent draws are taken per constraint and then sorted, rather
    than drawing and rejecting until the ordering happens to come out right.
    Sorting keeps the draw unbiased and never loops.

    Returns the five arrays that make up a `HysteronValve`.
    """
    states = np.zeros(N_valves)
    C0s    = np.zeros(N_valves)
    h_ms   = np.zeros(N_valves)
    h_ps   = np.zeros(N_valves)
    dCs    = np.zeros(N_valves)

    # Conductances: the larger draw becomes the open conductance, so that the
    # change on closing is negative.
    for i in range(N_valves):
        c1 = np.random.uniform(C_lower, C_upper)
        c2 = np.random.uniform(C_lower, C_upper)
        if c1 > c2:
            C0s[i] = c1
            dCs[i] = c2 - c1
        else:
            C0s[i] = c2
            dCs[i] = c1 - c2
        assert dCs[i] <= 0

    # Thresholds: the larger draw becomes the closing pressure, so that the
    # opening pressure sits below it and the hysteresis loop is non-empty.
    for i in range(N_valves):
        r1 = np.random.uniform(h_lower_bound, h_upper_bound)
        r2 = np.random.uniform(h_lower_bound, h_upper_bound)
        if r1 > r2:
            h_ps[i] = r1
            h_ms[i] = r2
        else:
            h_ps[i] = r2
            h_ms[i] = r1
        assert h_ms[i] <= h_ps[i]

    return states, h_ms, h_ps, C0s, dCs


def implement_parameters(params, valves, conds):
    """
    Point a `HysteronValve` at the slices of `params` it corresponds to.

    Note that this binds *views*, not copies: after calling it, mutating
    `params[i]` in place is visible through `valves`. The Adam solver relies
    on exactly that to make its finite-difference probe visible to the cost
    function.
    """
    N_valves = len(valves.state)
    valves.hm = params[:N_valves]
    valves.hp = params[N_valves:2 * N_valves]
    valves.C0 = params[2 * N_valves:3 * N_valves]
    valves.dC = params[3 * N_valves:4 * N_valves]
    conds     = params[4 * N_valves:]
    return valves, conds


def straighten_parameters(params, valves, conds):
    """
    Project a parameter vector back onto the physically meaningful region.

    Optimisers are free to propose anything, including negative conductances
    or a valve whose opening pressure sits above its closing pressure. Rather
    than reject such proposals, we map them to the nearest sensible ones.

    The trick is to work in midpoint/size coordinates: take the absolute value
    of each pair's midpoint and size, then rebuild the pair from them. That
    guarantees the orderings hm <= hp and dC <= 0 for *any* input, without
    ever having to discard a proposal, which matters because a rejected
    proposal would put a discontinuity in the cost landscape.

    Minimum sizes stop the optimiser from collapsing a valve into a plain
    conductor (a hysteresis loop of zero width, which can no longer switch)
    or a network into a short circuit.

    Returns the corrected `(params, valves, conds)`. `params` is written back
    in place, so the caller's vector stays consistent with `valves`.
    """
    min_h_mid  = 0.1
    min_c_mid  = 0.1
    min_h_size = 0.1
    min_c_size = 0.1

    N_valves = len(valves.state)
    h_mid     = np.abs((params[:N_valves] + params[N_valves:2 * N_valves]) / 2)
    h_size    = np.abs(params[:N_valves] - params[N_valves:2 * N_valves])
    c_mid     = np.abs(params[2 * N_valves:3 * N_valves]
                       + 0.5 * params[3 * N_valves:4 * N_valves])
    c_size    = np.abs(params[3 * N_valves:4 * N_valves])
    conds_arr = params[4 * N_valves:].copy()

    np.place(h_mid,  h_mid  < min_h_mid,  min_h_mid)
    np.place(c_mid,  c_mid  < min_c_mid,  min_c_mid)
    np.place(h_size, h_size < min_h_size, min_h_size)
    np.place(c_size, c_size < min_c_size, min_c_size)

    # Rebuild the four parameters from the corrected midpoints and sizes.
    hm  = h_mid - 0.5 * h_size
    hp  = h_mid + 0.5 * h_size
    C0s = c_mid + 0.5 * c_size      # open conductance, the larger of the two
    C1s = c_mid - 0.5 * c_size      # closed conductance

    # Keep everything strictly positive: a zero opening pressure or a zero
    # conductance would make the pressure solve singular.
    np.place(hm,        hm        < 0.5 * min_h_mid, 0.5 * min_h_mid)
    np.place(C1s,       C1s       < 0.5 * min_c_mid, 0.5 * min_c_mid)
    np.place(conds_arr, conds_arr < 0.5 * min_c_mid, 0.5 * min_c_mid)

    valves.hm = hm
    valves.hp = hp
    valves.C0 = C0s
    valves.dC = C1s - C0s

    params[:N_valves]                    = hm
    params[N_valves:2 * N_valves]        = hp
    params[2 * N_valves:3 * N_valves]    = C0s
    params[3 * N_valves:4 * N_valves]    = C1s - C0s
    params[4 * N_valves:]                = conds_arr

    return params, valves, conds_arr


def give_kick(params, valves, conds, ignore_list=()):
    """
    Randomly perturb every parameter to escape a flat region of the cost.

    Applied by the gradient descent, Adam and TPE solvers whenever the cost
    has failed to improve by more than some fraction over a window of steps.
    Each parameter is displaced by an independent uniform draw, and the result
    is straightened so the kicked point is still physical.

    `ignore_list` holds indices to leave untouched, for freezing parameters.
    """
    kicksize = 0.1
    for i in range(len(params)):
        if i in ignore_list:
            continue
        params[i] += np.random.uniform(-kicksize, kicksize)
    params, valves, conds = straighten_parameters(params, valves, conds)
    return valves, conds, params


def random_parameter_vector(N_valves, N_conds,
                            h_lower_bound, h_upper_bound,
                            C_lower_bound, C_upper_bound):
    """
    Draw a complete random parameter vector plus its matching `HysteronValve`.

    Convenience wrapper around `initialize_random_hysteron_valves` that also
    draws the conductor conductances and packs everything into the flat layout
    documented at the top of this module.

    The conductor draw comes first, matching the order the solvers have always
    used, so that a given seed reproduces the same starting point.
    """
    conds = np.random.uniform(C_lower_bound, C_upper_bound, N_conds)
    states, h_ms, h_ps, C0s, dCs = initialize_random_hysteron_valves(
        N_valves, h_lower_bound, h_upper_bound, C_lower_bound, C_upper_bound)
    params = np.concatenate((h_ms, h_ps, C0s, dCs, conds))
    valves = HysteronValve(states, h_ms, h_ps, C0s, dCs)
    return params, valves, conds
