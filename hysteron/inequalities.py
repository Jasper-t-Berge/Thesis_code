"""
Design inequalities and the cost function.

This is the heart of the inverse-design formulation. It turns "does this
network produce the right t-graph?" -- a yes/no question, useless to an
optimiser -- into "how far off is it?", a continuous number that can be
minimised.

The idea is that every transition in the target t-graph implies an ordering
between switching thresholds. If valve A is supposed to flip before valve B,
then A's threshold must be the lower of the two. Collect every such ordering
and the network reproduces the target t-graph exactly when all of them hold.

Where the orderings come from
-----------------------------
Take an up-transition from state S_initial to state S_final. The valves that
flip are the "critical" ones; every other valve has to stay put. That gives
two families of constraints:

  * The critical valve must flip before any other valve would. If some other
    open valve had a lower threshold, *it* would flip first and we would end
    up in a different state.
  * The state we land in must itself be stable. If a valve in S_final would
    immediately flip, the transition would not stop where the target says.

So each non-flipping valve is constrained twice, once in S_initial and once
in S_final. Whether the constraint involves its closing or its opening
threshold depends only on whether that valve is currently open or closed.

When several valves flip at once (a cascade), there is a third family: each
co-flipping valve must already be past its own threshold at the intermediate
state its partner leaves behind, otherwise the cascade stalls half way.

Encoding
--------
An inequality is stored as a (2, 3) block:

    ineq[0] = (sign, valve_index, state)   the threshold that must be smaller
    ineq[1] = (sign, valve_index, state)   the threshold that must be larger

`sign` records whether the entry refers to a closing threshold (+1) or an
opening one (-1). It is informational only: as explained in
`calculate_hysteron_thresholds_given_state`, a valve's state already fixes
which of its two thresholds applies, so the cost function never reads it. It
is kept because it makes generated inequalities readable when debugging.
"""

import numpy as np

from . import config
from .network import (
    calculate_hysteron_thresholds_given_state,
    convert_number_to_state,
)


def _flipped_bits(N_hyst, n1, n2):
    """Indices of the valves that differ between states n1 and n2."""
    xor = n1 ^ n2
    return [b for b in range(N_hyst) if (xor & (1 << (N_hyst - 1 - b)))]


def inequalities_up_transition(N_hyst, n1, n2):
    """
    Inequalities for the up-transition n1 -> n2, i.e. pressure being raised.

    `n1` is the initial state and `n2` the final one.

    The cascade branch builds its intermediate state by *setting* the critical
    bit, which assumes an up-transition only ever closes valves. That holds
    for all the target t-graphs used here, where up-transitions are strictly
    0 -> 1 and down-transitions strictly 1 -> 0.
    """
    flipped_bits = _flipped_bits(N_hyst, n1, n2)

    temp_ineq = np.zeros(shape=(4 * N_hyst * len(flipped_bits), 2, 3), dtype=int)
    state_arr = convert_number_to_state(n1, N_hyst)
    N_fill    = 0

    for critical_index in flipped_bits:

        # The intermediate state: n1 with only this valve flipped. It is a
        # state the network genuinely passes through, so the valves that are
        # supposed to stay put have to be stable there too. For a single-valve
        # transition it coincides with n2 and adds nothing, so it is only
        # brought in for cascades.
        n_inter = n1 | (1 << (N_hyst - 1 - critical_index))
        stable_states = [n1, n2]
        if len(flipped_bits) > 1:
            stable_states.append(n_inter)

        # --- Valves that must stay put, in each state the transition visits.
        for i in range(N_hyst):
            if i in flipped_bits:
                continue
            s = state_arr[i]

            for state in stable_states:
                if s == 0:
                    # Valve i is open, so the threshold that could move it is
                    # its closing threshold. The critical valve has to close
                    # first, in every state along the way.
                    temp_ineq[N_fill, 0] = (1, critical_index, n1)
                    temp_ineq[N_fill, 1] = (1, i, state)
                else:
                    # Valve i is closed, so what could move it is its opening
                    # threshold. The critical valve must close at a pressure
                    # *above* the one at which i would fall open, or the state
                    # could not hold together.
                    temp_ineq[N_fill, 1] = (1, critical_index, n1)
                    temp_ineq[N_fill, 0] = (-1, i, state)
                N_fill += 1

        # --- Cascades. By the time the critical valve flips, every other
        # co-flipping valve must already be past its own threshold in the
        # intermediate state, or the cascade stops there instead of running
        # all the way to n2.
        for other in flipped_bits:
            if other == critical_index:
                continue
            temp_ineq[N_fill, 0] = (1, other, n_inter)
            temp_ineq[N_fill, 1] = (1, critical_index, n1)
            N_fill += 1

    return temp_ineq[:N_fill]


def inequalities_down_transition(N_hyst, n1, n2):
    """
    Inequalities for the down-transition n1 -> n2, i.e. pressure being lowered.

    Mirror image of `inequalities_up_transition`. On the way down it is the
    *highest* threshold that is reached first, so every comparison flips
    direction.
    """
    flipped_bits = _flipped_bits(N_hyst, n1, n2)

    temp_ineq = np.zeros(shape=(4 * N_hyst * len(flipped_bits), 2, 3), dtype=int)
    state_arr = convert_number_to_state(n1, N_hyst)
    N_fill    = 0

    for critical_index in flipped_bits:

        # The intermediate state, clearing the critical bit since a
        # down-transition only ever opens valves. As above, it only adds
        # anything for a cascade.
        n_inter = n1 & ~(1 << (N_hyst - 1 - critical_index))
        stable_states = [n1, n2]
        if len(flipped_bits) > 1:
            stable_states.append(n_inter)

        # --- Valves that must stay put, in each state the transition visits.
        for i in range(N_hyst):
            if i in flipped_bits:
                continue
            s = state_arr[i]

            for state in stable_states:
                if s == 1:
                    # Valve i is closed: the critical valve must re-open
                    # before valve i would, on the way down.
                    temp_ineq[N_fill, 1] = (-1, critical_index, n1)
                    temp_ineq[N_fill, 0] = (-1, i, state)
                else:
                    # Valve i is open: the critical valve must re-open at a
                    # pressure below the one at which i would close.
                    temp_ineq[N_fill, 0] = (-1, critical_index, n1)
                    temp_ineq[N_fill, 1] = (1, i, state)
                N_fill += 1

        # --- Cascades. Each other co-flipping valve must already be below its
        # own threshold in the intermediate state.
        for other in flipped_bits:
            if other == critical_index:
                continue
            temp_ineq[N_fill, 0] = (-1, critical_index, n1)
            temp_ineq[N_fill, 1] = (-1, other, n_inter)
            N_fill += 1

    return temp_ineq[:N_fill]


def build_inequalities(N_valves, up_trans, down_trans):
    """
    Stack the inequalities for every transition in a target t-graph.

    Returns a single (n, 2, 3) array, which is what the cost function consumes.
    """
    ineq_array = np.array([])

    for trans in up_trans:
        temp_ineq = inequalities_up_transition(N_valves, trans[0], trans[1])
        ineq_array = np.vstack([ineq_array, temp_ineq]) if ineq_array.size else temp_ineq

    for trans in down_trans:
        temp_ineq = inequalities_down_transition(N_valves, trans[0], trans[1])
        ineq_array = np.vstack([ineq_array, temp_ineq]) if ineq_array.size else temp_ineq

    return ineq_array


def cost_function_system(ineqs, valves, conds, boundary_pressure_template,
                         incidence_matrix, epsilon=config.EPSILON):
    """
    Total violation of the design inequalities.

        cost = sum over inequalities of  max(0, P_smaller - P_larger + epsilon)

    A satisfied inequality contributes nothing; a violated one contributes the
    size of its violation. Summing the magnitudes rather than just counting
    failures is what makes the landscape continuous, and it is why the
    optimisers can make progress at all: they can tell a nearly correct
    network from a badly wrong one, instead of only being told that both are
    wrong.

    The landscape is continuous but not smooth. There is a kink wherever a
    constraint crosses from violated to satisfied, which is why the
    derivative-free and non-smoothness-tolerant methods do well on it.

    The cost is zero exactly when every inequality holds, i.e. when the
    network produces the target t-graph. See `config.EPSILON` for the margin.

    Thresholds are computed once per distinct state and cached, since the same
    state appears in many inequalities and the pressure solve is by far the
    most expensive part of the evaluation.
    """
    all_states = np.unique(ineqs[:, :, 2].flatten())
    N_valves   = len(valves.state)
    N_states   = len(all_states)

    all_thresholds = np.zeros(shape=(N_states, N_valves))
    for i in range(N_states):
        all_thresholds[i] = calculate_hysteron_thresholds_given_state(
            all_states[i], valves, conds,
            boundary_pressure_template, incidence_matrix)

    dict_t = dict(zip(all_states.tolist(), all_thresholds.tolist()))

    cost = 0
    for i in range(len(ineqs)):
        # ineqs[i, 0] must be the smaller threshold, ineqs[i, 1] the larger.
        val = (dict_t[ineqs[i, 0, 2]][ineqs[i, 0, 1]]
               - dict_t[ineqs[i, 1, 2]][ineqs[i, 1, 1]]
               + epsilon)
        if val > 0:
            cost += val

    return cost
