"""
Drawing t-graphs.

Purely for inspecting a solution by eye; nothing here feeds back into the
optimisation. States are laid out by how many valves are closed, so the
all-open state sits at the bottom, the all-closed state at the top, and each
horizontal band holds the states with the same number of closed valves.
Up-transitions are drawn blue and down-transitions red.
"""

from math import comb

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .network import convert_number_to_state


def plot_t_graph(t_graph_array, blockbool=False, spinshow=False):
    """
    Draw a t-graph produced by `find_t_graph_with_thresholds`.

    Set `spinshow` to label nodes with their binary valve states rather than
    just the packed integer.
    """
    states  = t_graph_array[:, 0].astype(int)
    max_val = int(np.max(states))
    # Number of valves needed to represent the largest state seen. bit_length
    # is exact at powers of two, where ceil(log2(x)) would be off by one.
    N_edges = max(1, max_val.bit_length())

    coordinate_array = coordinates_t_graph(states, N_edges)

    G = nx.DiGraph()
    for i in range(len(t_graph_array)):
        coor = None
        for j in range(len(coordinate_array)):
            if coordinate_array[j, 0] == t_graph_array[i, 0]:
                coor = (coordinate_array[j, 1], coordinate_array[j, 2])
                break
        if coor is None:
            print(f"error: coordinate of node {t_graph_array[i, 0]} not found")
            continue
        G.add_node(t_graph_array[i, 0], pos=coor)

    # Columns 1 and 2 hold the down- and up-transition targets; -1 means the
    # state has no transition in that direction.
    for i in range(len(t_graph_array)):
        if t_graph_array[i, 1] >= 0:
            G.add_edge(t_graph_array[i, 0], t_graph_array[i, 1], color='r')  # down
        if t_graph_array[i, 2] >= 0:
            G.add_edge(t_graph_array[i, 0], t_graph_array[i, 2], color='b')  # up

    colors = nx.get_edge_attributes(G, 'color').values()
    plt.figure()
    plt.title('t-graph of system')

    if spinshow:
        G = add_spin_labels_to_nodes(G, t_graph_array, N_edges)
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx(G, with_labels=True, edge_color=colors, pos=pos,
                         node_size=500, node_color="#b8e1fc", font_size=8)
        plt.show(block=blockbool)
        return

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx(G, with_labels=True, edge_color=colors, pos=pos,
                     node_size=200, node_color="#b8e1fc", font_size=8)
    plt.show(block=blockbool)


def add_spin_labels_to_nodes(G, t_graph, N_edges):
    """Relabel nodes as e.g. '1011\\n(11)' instead of just '11'."""
    labels = {}
    for i in range(len(t_graph)):
        state = convert_number_to_state(int(t_graph[i, 0]), N_edges)
        lab = "".join(str(int(s)) for s in state) + f"\n({int(t_graph[i, 0])})"
        labels[int(t_graph[i, 0])] = lab
    return nx.relabel_nodes(G, labels, copy=True)


def coordinates_t_graph(states, N_edges):
    """
    Assign each state an (x, y) position.

    y is the number of closed valves, so transitions generally run upwards.
    x spreads the states within a band, with the width of each band scaled by
    how many states it could hold. That gives the layout its diamond shape:
    narrow at the top and bottom, where few states share a valve count, and
    widest in the middle.

    Returns an (n, 3) array of (state, x, y).
    """
    N_groups              = N_edges + 1
    N_states              = len(states)
    vert_per_element      = 1 / N_groups
    max_states_all_groups = comb(N_edges, int(np.floor(N_edges / 2)))

    # Pair each state with its number of closed valves.
    magnetisation_states = np.zeros(shape=(N_states, 2), dtype=int)
    magnetisation_states[:, 0] = states
    for i in range(N_states):
        magnetisation_states[i, 1] = np.sum(
            convert_number_to_state(int(states[i]), N_edges))

    coordinate_array = np.zeros(shape=(N_states, 3))
    index = 0

    for i in range(N_groups):
        max_states_group = comb(N_edges, i)
        assert max_states_group <= max_states_all_groups

        # Collect the states belonging to this band.
        group_array = np.zeros(max_states_group, dtype=int)
        states_in_group_index = 0
        for j in range(N_states):
            if magnetisation_states[j, 1] == i:
                group_array[states_in_group_index] = magnetisation_states[j, 0]
                states_in_group_index += 1

        if states_in_group_index == 0:
            continue

        start_x = -0.5 * (max_states_group / max_states_all_groups)
        end_x   =  0.5 * (max_states_group / max_states_all_groups)

        group_array    = np.sort(group_array[:states_in_group_index])
        fraction_array = np.zeros(states_in_group_index)
        for k in range(states_in_group_index):
            state = convert_number_to_state(group_array[k], N_edges)
            fraction_array[k] = coordinate_horizontal_fraction_within_group(
                state, N_edges, i)

        assert len(group_array) == len(fraction_array)

        for j in range(states_in_group_index):
            coordinate_array[index, 0] = group_array[j]
            coordinate_array[index, 1] = start_x + (end_x - start_x) * fraction_array[j]
            coordinate_array[index, 2] = vert_per_element * i
            index += 1

    return coordinate_array


def coordinate_horizontal_fraction_within_group(state, N_valves, N_on):
    """
    Where within its band a state sits, as a fraction in [0, 1].

    Orders states within a band by their combinatorial rank, so that states
    sharing a valve count are spread out evenly and deterministically rather
    than piling up on top of each other.
    """
    if N_on == 0 or N_on == N_valves:
        return 0.5

    total_pos    = comb(N_valves, N_on)
    num_larger   = 0
    N_on_current = N_on

    for i in range(N_valves):
        if state[i] == 0:
            num_larger += comb(N_valves - i - 1, N_on_current - 1)
        if state[i] == 1:
            N_on_current -= 1
        if N_on_current == 0:
            break

    return num_larger / (total_pos - 1)


def print_hysteron(hyst, i):
    """Print one valve's parameters."""
    print(f'state = {hyst.state[i]}\th- = {round(hyst.hm[i], 10)}\t'
          f'h+ = {round(hyst.hp[i], 10)}\tC0 = {round(hyst.C0[i], 10)}\t'
          f'dC = {round(hyst.dC[i], 10)}')


def print_all_hysterons(states):
    """Print every valve's parameters, one per line."""
    for i in range(len(states.C0)):
        print_hysteron(states, i)


def print_representation_hysterons(hysts):
    """Print valve arrays in a form that can be pasted back into Python."""
    print("states = ", repr(hysts.state))
    print("hps =",    repr(hysts.hp))
    print("hms =",    repr(hysts.hm))
    print("C0s =",    repr(hysts.C0))
    print("dCs =",    repr(hysts.dC))
