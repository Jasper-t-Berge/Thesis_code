"""
The physical network: valves, topologies, and the pressure solve.

  * `HysteronValve` holds the four parameters of a fluidic hysteron valve.
  * `initialize_geometry` defines the three network topologies studied here.
  * `calculate_pressure_hysteron_and_conductor_network` solves Kirchhoff's
    equations for the pressure drop across every edge.
  * `find_threshold_up_transition` / `find_threshold_down_transition` find
    which valve flips next, and at what source pressure.
"""

import time

import numpy as np

from . import config


class HysteronValve:
    """
    A bank of fluidic hysteron valves, stored as parallel arrays.

    One instance describes *all* valves in a network at once; element i of
    each array refers to valve i. Each valve has four parameters:

        hp   closing pressure h+   -- the valve semi-closes when the pressure
                                      drop across it rises above this
        hm   opening pressure h-   -- it re-opens when the drop falls back
                                      below this, and hm <= hp, so the two
                                      thresholds enclose a hysteresis loop
        C0   open conductance
        dC   conductance change on closing, always <= 0, so the closed
             conductance C0 + dC can only be lower than the open one

    `state` is the binary state of each valve: 0 = open, 1 = closed. It is the
    only thing that changes as pressure is swept; the four parameters are what
    the inverse-design problem searches over.
    """

    def __init__(self, state, hm, hp, C0, dC):
        self.state = state
        self.hm    = hm
        self.hp    = hp
        self.C0    = C0
        self.dC    = dC


def set_RNG_seed(seed=-1):
    """Seed numpy's global RNG. Passing -1 derives a seed from the clock."""
    if seed != -1:
        print('seed:', seed, "\n")
        np.random.seed(seed)
    else:
        starting_seed = int(time.time() * 10000) % 100000000
        print('seed:', starting_seed, "\n")
        np.random.seed(starting_seed)


def initialize_geometry(topology='wheatstone'):
    """
    Return the edge list and size of one of the three studied topologies.

    Every topology follows the same convention, which the rest of the code
    relies on:

      * Node 0 is the sink, node 1 is the source, nodes 2+ are internal.
      * Edges 0 .. N_valves-1 are hysteron valves.
      * Edges N_valves .. are plain conductors, which cannot switch.

    All three topologies use 4 valves plus 1 conductor. Keeping the count
    identical is what makes them comparable: each supports target behaviours
    spanning the same 4 valve states, and each has the same 4*4 + 1 = 17
    dimensional parameter space. Making the 5th edge a fixed conductor rather
    than a fifth valve is a deliberate choice to keep that space small.

    Returns
    -------
    edges     : (N_edges, 2) int array of (from_node, to_node)
    N_valves  : number of switchable valves
    N_conds   : number of fixed conductors
    N_nodes   : total node count
    """

    if topology == 'series':
        # Four valves in a single chain, closed by the conductor. The coupling
        # here is negative: one valve closing throttles the whole chain, which
        # lowers the drop across the others and makes them harder to close.
        N_valves = 4
        N_conds  = 1
        N_edges  = N_valves + N_conds
        N_nodes  = 6
        edges = np.zeros(shape=(N_edges, 2), dtype=int)
        edges[0] = (1, 2)   # valve 0   source -> n2
        edges[1] = (2, 3)   # valve 1
        edges[2] = (3, 4)   # valve 2
        edges[3] = (4, 5)   # valve 3
        edges[4] = (5, 0)   # conductor n5 -> sink

    elif topology == 'series-parallel':
        # Valves 0-2 sit in parallel between the source and the single
        # internal node; valve 3 and the conductor run in parallel from there
        # to the sink. The parallel branches couple positively (one valve
        # closing diverts flow into its neighbours, helping them close) while
        # the two stages in series couple negatively. Mixing the two is what
        # lets this topology support cascading transitions.
        N_valves = 4
        N_conds  = 1
        N_edges  = N_valves + N_conds
        N_nodes  = 3
        edges = np.zeros(shape=(N_edges, 2), dtype=int)
        edges[0] = (1, 2)   # valve 0   source -> n2
        edges[1] = (1, 2)   # valve 1   source -> n2
        edges[2] = (1, 2)   # valve 2   source -> n2
        edges[3] = (2, 0)   # valve 3   n2 -> sink
        edges[4] = (2, 0)   # conductor n2 -> sink

    elif topology == 'wheatstone':
        # A bridge circuit: valve 1 spans the two internal nodes, and the
        # conductor is the 5th edge. The effect of one valve flipping on the
        # others is not obvious here, and that richer coupling is what gives
        # this topology the most complex reachable behaviour.
        N_valves = 4
        N_conds  = 1
        N_edges  = N_valves + N_conds
        N_nodes  = 4
        edges = np.zeros(shape=(N_edges, 2), dtype=int)
        edges[0] = (1, 3)   # valve 0   source -> n3
        edges[1] = (3, 2)   # valve 1   n3 -> n2   (the bridge)
        edges[2] = (2, 0)   # valve 2   n2 -> sink
        edges[3] = (3, 0)   # valve 3   n3 -> sink
        edges[4] = (1, 2)   # conductor source -> n2

    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            "Choose from: 'series', 'series-parallel', 'wheatstone'."
        )

    return edges, N_valves, N_conds, N_nodes


def build_incidence_matrix(edges, N_edges, N_nodes):
    """
    Build the incidence matrix D that encodes the network's wiring.

    Rows are edges, columns are nodes. The entry is +1 if the edge leaves the
    node, -1 if it enters, and 0 otherwise.
    """
    incidence_matrix = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        incidence_matrix[i, edges[i, 0]] =  1
        incidence_matrix[i, edges[i, 1]] = -1
    return incidence_matrix


def convert_number_to_state(num, length):
    """Integer -> binary state array, most significant bit first."""
    string = bin(num)[2:].zfill(length)
    return np.array(list(string), dtype=int)


def convert_state_to_number(state):
    """Binary state array -> integer. Inverse of `convert_number_to_state`."""
    num = 0
    for i in range(len(state)):
        num += (2**i) * state[-i - 1]
    return num


def calculate_pressure_hysteron_and_conductor_network(
        hysts, conds, boundary_pressure, incidence_matrix):
    """
    Solve Kirchhoff's equations for the pressure drop across every edge.

    Flow through an edge is conductance times pressure drop, and flow is
    conserved at every internal node. Writing both conditions together gives a
    linear system in the unknown internal node pressures:

        A_nn P_internal = -A_nb P_boundary

    where A_nn = Dn' C Dn describes how the internal nodes are coupled to each
    other and A_nb = Dn' C Db how the fixed boundary pressures drive them.
    Solving it and multiplying back through D gives the drop across each edge.

    An open valve (state 0) contributes conductance C0, a closed one (state 1)
    contributes C0 + dC. Conductors never switch, so they are appended
    unchanged.

    Returns
    -------
    pressure_edges : drop across each edge, ordered valves first then
                     conductors. With `symmetric_flag` set these are
                     magnitudes, hence fractions of the source pressure
                     lying in [0, 1].
    """
    Nb = len(boundary_pressure)
    weights_hyst = hysts.C0 + hysts.dC * hysts.state
    weights = np.concatenate((weights_hyst, conds))

    # Split the incidence matrix into its boundary and internal columns.
    Db = incidence_matrix[:, :Nb]
    Dn = incidence_matrix[:, Nb:]

    C_matrix = np.diag(weights)
    Anb = Dn.T @ C_matrix @ Db
    Ann = Dn.T @ C_matrix @ Dn

    internal_pressure = -np.linalg.inv(Ann) @ Anb @ boundary_pressure

    pressure_nodes = np.concatenate((boundary_pressure, internal_pressure))
    pressure_edges = incidence_matrix @ pressure_nodes

    # Read through `config` on every call rather than captured at import, so
    # setting `config.symmetric_flag` at runtime actually takes effect.
    if config.symmetric_flag:
        pressure_edges = np.abs(pressure_edges)

    return pressure_edges


def find_threshold_up_transition(valves, pressure_edges, index_flag=True):
    """
    Find which valve flips first as the source pressure is *raised*.

    A valve flips once the drop across it reaches its intrinsic threshold. At
    unit boundary pressure the drop is a fixed fraction of the source
    pressure, so the source pressure needed to flip valve i is

        threshold_i = h_i / pressure_drop_i

    The valve that flips first on the way up is whichever has the lowest such
    threshold. Note this is set by the network as much as by the valve: a
    valve on a high-conductance path takes a larger share of the total drop
    and so flips at a lower source pressure.

    Returns the threshold, and optionally the index of the valve that flips.
    An index of -1 with a threshold of +inf means no valve can flip upwards.
    """
    lowest_val = np.inf
    index      = -1

    for i in range(len(valves.state)):
        if valves.state[i] == 0 and pressure_edges[i] > 0:
            # Open valve pushed towards closing: it hits h+ on the way up.
            val = valves.hp[i] / pressure_edges[i]
        elif valves.state[i] == 1 and pressure_edges[i] < 0:
            # Only reachable with symmetric_flag off, when the flow through a
            # closed valve is reversed.
            val = valves.hm[i] / pressure_edges[i]
        else:
            # This valve cannot flip upwards from its current state.
            continue

        if val < lowest_val:
            lowest_val = val
            index      = i

    if index_flag:
        return lowest_val, index
    return lowest_val


def find_threshold_down_transition(valves, pressure_edges, index_flag=True):
    """
    Find which valve flips first as the source pressure is *lowered*.

    Mirror image of `find_threshold_up_transition`: a closed valve re-opens
    once the drop across it falls below h-, and the valve that flips first on
    the way down is the one with the *highest* threshold.

    An index of -1 with a threshold of -inf means no valve can flip downwards.
    """
    highest_val = -np.inf
    index       = -1

    for i in range(len(valves.state)):
        if valves.state[i] == 1 and pressure_edges[i] > 0:
            # Closed valve relaxing: it hits h- on the way down.
            val = valves.hm[i] / pressure_edges[i]
        elif valves.state[i] == 0 and pressure_edges[i] < 0:
            # Only reachable with symmetric_flag off.
            val = valves.hp[i] / pressure_edges[i]
        else:
            # This valve cannot flip downwards from its current state.
            continue

        if val > highest_val:
            highest_val = val
            index       = i

    if index_flag:
        return highest_val, index
    return highest_val


def calculate_hysteron_thresholds_given_state(
        num_state, valves, conds, boundary_pressure_template, incidence_matrix):
    """
    Switching threshold of every valve in one given global state.

    In a fixed state each valve has exactly one threshold that matters: an
    open valve can only close, so h+ applies, and a closed valve can only
    open, so h- applies. The state therefore determines which threshold is
    meant, which is why the design inequalities never have to say so.

    Returns an array of length N_valves, indexed by valve.
    """
    N_valves = len(valves.state)
    valves.state = convert_number_to_state(num_state, N_valves)
    pressures = calculate_pressure_hysteron_and_conductor_network(
        valves, conds, boundary_pressure_template, incidence_matrix)

    # A valve carrying no pressure drop can never reach its threshold. The
    # division below yields +inf for it, which is the correct answer and is
    # handled by the callers, so the warning is silenced rather than avoided.
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(N_valves):
            if valves.state[i] == 0:
                pressures[i] = valves.hp[i] / pressures[i]
            else:
                pressures[i] = valves.hm[i] / pressures[i]

    return pressures[:N_valves]
