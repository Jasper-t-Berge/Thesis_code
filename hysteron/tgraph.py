"""
Transition graphs: the target behaviours, and how to read one off a network.

A t-graph is a state machine over global valve states. Each node is a state
(the binary states of all valves, packed into an integer) and each edge is a
transition triggered by raising or lowering the source pressure. It is the
object that captures what a network *does*, independent of the parameter
values that happen to produce it.

Two directions of travel live here:

  * `initialize_tgraph` gives the *target* behaviours we design towards.
  * `find_t_graph_with_thresholds` goes the other way, extracting the t-graph
    a concrete network actually produces. That is the forward problem, used
    for inspecting and plotting a solution rather than by the optimisers.
"""

import numpy as np

from .network import (
    calculate_pressure_hysteron_and_conductor_network,
    convert_number_to_state,
    convert_state_to_number,
    find_threshold_down_transition,
    find_threshold_up_transition,
)


def initialize_tgraph(tgraph='MPO'):
    """
    Return the target transition lists for one of the three studied behaviours.

    Each transition is a pair [from_state, to_state] of packed integers over
    4 valves, so state 11 = binary 1011 means valves 0, 2 and 3 closed. All
    three targets span 4 valve states, which is what makes them comparable
    across topologies.

    Only the transitions that define the behaviour are listed; the rest of the
    t-graph is left free for the optimiser to fill in however it likes.

    Returns
    -------
    up_trans   : transitions taken when the source pressure is raised
    down_trans : transitions taken when it is lowered
    """

    if tgraph == 'Preisach':
        # The non-interacting case: valves flip in a fixed order and every
        # transition is a single flip. The simplest behaviour to realise.
        up_trans   = [[0, 1], [1, 3], [3, 7], [7, 15], [11, 15]]
        down_trans = [[15, 11], [11, 9], [9, 8], [8, 0], [7, 3]]

    elif tgraph == 'Avalanche':
        # The transition 3 -> 15 is the avalanche: two valves flip together
        # for a single increase in pressure, because the first one closing
        # pushes the second past its own threshold immediately.
        up_trans   = [[0, 1], [1, 3], [3, 15]]
        down_trans = [[15, 14], [14, 6], [6, 4], [4, 0]]

    elif tgraph == 'MPO':
        # A multi-periodic orbit. One "pulse" is a rise followed by a fall,
        # and the system returns to its starting state only after several
        # pulses rather than after one, which is what allows it to count.
        up_trans   = [[0, 1], [1, 5], [5, 7], [2, 3], [3, 11]]
        down_trans = [[7, 3], [3, 2], [11, 10], [10, 8], [8, 0]]

    else:
        raise ValueError(
            f"Unknown tgraph '{tgraph}'. "
            "Choose from: 'Preisach', 'Avalanche', 'MPO'."
        )

    return up_trans, down_trans


def check_for_avalanches(valves, conds, boundary_pressure_template,
                         incidence_matrix, P_val, num_avalanches=100):
    """
    Relax a network to a stable state after a valve has just flipped.

    A flip changes the conductances, which redistributes the pressure, which
    can push another valve past its own threshold. That cascade is an
    avalanche. Because the model treats both the flip and the flow
    redistribution as instantaneous, the whole cascade counts as a single
    transition rather than several.

    Iterates until the state is stable at pressure `P_val`, flipping the most
    unstable valve each round, or until `num_avalanches` rounds have passed.
    """
    for _ in range(num_avalanches):

        pressure_edges = calculate_pressure_hysteron_and_conductor_network(
            valves, conds, boundary_pressure_template, incidence_matrix)

        down_val, down_index = find_threshold_down_transition(
            valves, pressure_edges, index_flag=True)
        up_val, up_index = find_threshold_up_transition(
            valves, pressure_edges, index_flag=True)

        # Stable: the applied pressure sits strictly between the nearest
        # up-threshold and the nearest down-threshold, so nothing flips.
        if down_val < P_val < up_val:
            return valves

        # Otherwise flip whichever valve is furthest past its threshold, and
        # look again.
        error_up   = P_val - up_val
        error_down = down_val - P_val

        if error_up > error_down:
            valves.state[up_index] = (valves.state[up_index] + 1) % 2
        else:
            valves.state[down_index] = (valves.state[down_index] + 1) % 2

    print("not enough avalanche cycles or self-loop present")
    return valves


def calculate_state_and_threshold_increasing_P(
        valves, conds, boundary_pressure_template, incidence_matrix, state_int):
    """
    From `state_int`, raise the pressure until something flips.

    Returns the resulting stable state and the source pressure at which the
    transition happened, or (-1, +inf) if no valve can flip upwards at all.
    """
    valves.state = convert_number_to_state(state_int, len(valves.state))

    p_edges = calculate_pressure_hysteron_and_conductor_network(
        valves, conds, boundary_pressure_template, incidence_matrix)

    threshold, unstable_i = find_threshold_up_transition(
        valves, p_edges, index_flag=True)

    if threshold == np.inf:
        return -1, np.inf

    valves.state[unstable_i] = (valves.state[unstable_i] + 1) % 2
    current_valves = check_for_avalanches(
        valves, conds, boundary_pressure_template, incidence_matrix, threshold)

    return convert_state_to_number(current_valves.state), threshold


def calculate_state_and_threshold_decreasing_P(
        valves, conds, boundary_pressure_template, incidence_matrix, state_int):
    """
    From `state_int`, lower the pressure until something flips.

    Returns the resulting stable state and the source pressure at which the
    transition happened, or (-1, -inf) if no valve can flip downwards.
    """
    valves.state = convert_number_to_state(state_int, len(valves.state))

    p_edges = calculate_pressure_hysteron_and_conductor_network(
        valves, conds, boundary_pressure_template, incidence_matrix)

    threshold, unstable_i = find_threshold_down_transition(
        valves, p_edges, index_flag=True)

    if threshold == -np.inf:
        return -1, -np.inf

    valves.state[unstable_i] = (valves.state[unstable_i] + 1) % 2
    current_valves = check_for_avalanches(
        valves, conds, boundary_pressure_template, incidence_matrix, threshold)

    return convert_state_to_number(current_valves.state), threshold


def find_t_graph_with_thresholds(valves, conds, boundary_pressure_template,
                                 incidence_matrix):
    """
    Extract the full t-graph produced by a concrete network.

    Explores outwards from the all-open state, following up- and
    down-transitions until no new state appears. This is the forward problem:
    useful for checking and plotting what a solution actually does, but not
    used by the optimisers, which work from the inequalities instead.

    Returns
    -------
    (n, 5) array sorted by state, with columns:
        0  state
        1  state reached by lowering the pressure
        2  state reached by raising the pressure
        3  threshold of the down-transition
        4  threshold of the up-transition
    """
    N_valves   = len(valves.state)
    max_states = 2 ** N_valves
    t_graph    = np.zeros(shape=(max_states, 5)) - 1
    t_graph[0, 0] = 0
    N_fill     = 1
    fill_index = 0

    for _ in range(max_states):
        state_int = int(t_graph[fill_index, 0])

        t_graph[fill_index, 1], t_graph[fill_index, 3] = \
            calculate_state_and_threshold_decreasing_P(
                valves, conds, boundary_pressure_template, incidence_matrix,
                state_int)
        t_graph[fill_index, 2], t_graph[fill_index, 4] = \
            calculate_state_and_threshold_increasing_P(
                valves, conds, boundary_pressure_template, incidence_matrix,
                state_int)

        # Queue up any newly discovered states. Unfilled rows hold -1, which
        # is also the "no transition" marker, so dead ends are never queued.
        if t_graph[fill_index, 1] not in t_graph[:, 0]:
            t_graph[N_fill] = t_graph[fill_index, 1]
            N_fill += 1
        if t_graph[fill_index, 2] not in t_graph[:, 0]:
            t_graph[N_fill] = t_graph[fill_index, 2]
            N_fill += 1

        fill_index += 1
        if fill_index >= N_fill:
            break

    t_graph = t_graph[:N_fill, :]
    return t_graph[t_graph[:, 0].argsort()]


def dict_orbits_of_t_graph(t_graph, N_edges, dead_end=2):
    """
    Follow each state's up- and down-orbit to its end.

    An "orbit" is the chain of states passed through by repeatedly raising
    (or repeatedly lowering) the pressure, until the network either saturates
    or hits a dead end. Used by the return-point-memory check below.
    """
    min_state = 0
    max_state = 2 ** N_edges - 1

    # Column 2 holds up-transitions and column 1 down-transitions, matching
    # the layout documented in find_t_graph_with_thresholds.
    dict_up   = dict(zip(t_graph[:, 0].tolist(), t_graph[:, 2].tolist()))
    dict_down = dict(zip(t_graph[:, 0].tolist(), t_graph[:, 1].tolist()))

    up_orbits   = np.zeros(shape=(len(t_graph), N_edges + dead_end), dtype=int) - 1
    down_orbits = np.zeros(shape=(len(t_graph), N_edges + dead_end), dtype=int) - 1

    for i in range(len(t_graph)):
        up_orbits[i, 0]   = t_graph[i, 0]
        down_orbits[i, 0] = t_graph[i, 0]

        for j in range(N_edges + dead_end - 1):
            up_orbits[i, j + 1] = dict_up[up_orbits[i, j]]
            if up_orbits[i, j + 1] == -1 or up_orbits[i, j + 1] == max_state:
                break
        for j in range(N_edges + dead_end - 1):
            down_orbits[i, j + 1] = dict_down[down_orbits[i, j]]
            if down_orbits[i, j + 1] == -1 or down_orbits[i, j + 1] == min_state:
                break

    up_orbits   = dict(zip(up_orbits[:, 0].tolist(),   up_orbits[:, 1:].tolist()))
    down_orbits = dict(zip(down_orbits[:, 0].tolist(), down_orbits[:, 1:].tolist()))
    return up_orbits, down_orbits


def check_loop_RPM(down, up, up_orbits, down_orbits):
    """Check return-point memory for the single loop between `down` and `up`."""
    for s in up_orbits[down]:
        if s == up:
            break
        if not (down in down_orbits[s]):
            return False

    for s in down_orbits[up]:
        if s == down:
            break
        if not (up in up_orbits[s]):
            return False

    return True


def check_t_graph_loop_RPM(t_graph, N_edges):
    """
    Test whether a t-graph has return-point memory throughout.

    Return-point memory means that after a pressure excursion the network
    comes back to the state it started from, rather than drifting to a
    different one. A diagnostic for inspecting solutions; not part of the
    cost function.
    """
    up_orbits, down_orbits = dict_orbits_of_t_graph(t_graph, N_edges)

    for j in range(len(t_graph)):
        current_state    = t_graph[j, 0]
        current_up_orbit = up_orbits[current_state]

        for s in current_up_orbit:
            if s == -1:
                break
            if current_state in down_orbits[s]:
                if not check_loop_RPM(current_state, s, up_orbits, down_orbits):
                    return False
    return True
