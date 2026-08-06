"""
Simulation and inverse design of fluidic hysteron valve networks.

The package is split into five parts:

    config        global constants: boundary conditions, cost margin
    network       valves, topologies, and the Kirchhoff pressure solve
    tgraph        target behaviours, and reading a t-graph off a network
    inequalities  design inequalities and the cost function
    parameters    parameter layout, initialisation, kicks
    optimize      the CMA-ES search, shared by both experiments
    plotting      t-graph visualisation

The optimisers that search this parameter space live outside the package, in
`compare_optimizers/` and `topology_sweep/`.
"""

from . import config
from .inequalities import (
    build_inequalities,
    cost_function_system,
    inequalities_down_transition,
    inequalities_up_transition,
)
from .optimize import cma_search, new_cma_es
from .network import (
    HysteronValve,
    build_incidence_matrix,
    calculate_hysteron_thresholds_given_state,
    calculate_pressure_hysteron_and_conductor_network,
    convert_number_to_state,
    convert_state_to_number,
    find_threshold_down_transition,
    find_threshold_up_transition,
    initialize_geometry,
    set_RNG_seed,
)
from .parameters import (
    give_kick,
    implement_parameters,
    initialize_random_hysteron_valves,
    random_parameter_vector,
    straighten_parameters,
)
from .tgraph import (
    check_t_graph_loop_RPM,
    find_t_graph_with_thresholds,
    initialize_tgraph,
)

__all__ = [
    "config",
    "HysteronValve",
    "build_incidence_matrix",
    "build_inequalities",
    "calculate_hysteron_thresholds_given_state",
    "calculate_pressure_hysteron_and_conductor_network",
    "check_t_graph_loop_RPM",
    "convert_number_to_state",
    "convert_state_to_number",
    "cma_search",
    "cost_function_system",
    "find_t_graph_with_thresholds",
    "find_threshold_down_transition",
    "find_threshold_up_transition",
    "give_kick",
    "implement_parameters",
    "inequalities_down_transition",
    "inequalities_up_transition",
    "initialize_geometry",
    "initialize_random_hysteron_valves",
    "initialize_tgraph",
    "new_cma_es",
    "random_parameter_vector",
    "set_RNG_seed",
    "straighten_parameters",
]
