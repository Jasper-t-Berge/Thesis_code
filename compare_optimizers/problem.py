"""
The one inverse-design problem that all four optimisers are benchmarked on.

Every solver in this folder imports this module, so they all attack an
identical problem and any difference in their results is down to the optimiser
rather than the setup.

The problem is fixed as:

  * The Wheatstone topology, whose richer coupling between valves makes it the
    hardest of the three and the only one able to realise the MPO behaviour.
    Its 5th edge is a plain conductor, giving a 17-dimensional search space.
  * The MPO target t-graph. It is known to be realisable on this topology, and
    it is demanding enough that the methods actually separate: on a trivial
    target they would all succeed immediately and there would be nothing to
    compare.
  * A budget of 10,000 cost function evaluations per run. Large enough for the
    stronger methods to succeed on most seeds, tight enough that differences
    in efficiency still show up.
"""

import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

from hysteron import (                                        # noqa: E402
    build_incidence_matrix,
    build_inequalities,
    config,
    initialize_geometry,
    initialize_tgraph,
)

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
TOPOLOGY      = 'wheatstone'
TARGET_TGRAPH = 'MPO'

# Ranges the random starting points are drawn from.
H_LOWER_BOUND, H_UPPER_BOUND = 0.1, 1.0
C_LOWER_BOUND, C_UPPER_BOUND = 0.5, 5.0

# Cost function evaluations allowed per run.
EVAL_BUDGET = 10_000

# --------------------------------------------------------------------------
# Derived geometry
# --------------------------------------------------------------------------
edges, N_valves, N_conds, N_nodes = initialize_geometry(topology=TOPOLOGY)
N_edges  = N_valves + N_conds
N_params = N_valves * 4 + N_conds        # 17

incidence_matrix = build_incidence_matrix(edges, N_edges, N_nodes)

boundary_pressure_template = config.boundary_pressure_template

# --------------------------------------------------------------------------
# Design inequalities for the target t-graph
# --------------------------------------------------------------------------
up_trans, down_trans = initialize_tgraph(tgraph=TARGET_TGRAPH)
ineq_array = build_inequalities(N_valves, up_trans, down_trans)
