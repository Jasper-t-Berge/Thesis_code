"""
Global constants shared by every script in this project.

These are the quantities fixed once for the whole study, as opposed to the
per-experiment settings (topology, target t-graph, optimiser hyperparameters)
which live with the experiments themselves.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
# Every network has exactly one source and one sink. Nodes are numbered so
# that the boundary nodes come first: node 0 is the sink, node 1 is the
# source, and everything from node 2 onwards is internal.
N_source = 1
N_sink   = 1
Nb       = N_source + N_sink          # number of boundary nodes

# Networks are solved at unit boundary pressure, (P_source, P_sink) = (1, 0).
# Working at unit pressure means the drop across a valve comes out as the
# *fraction* of the source pressure that valve experiences, which is what
# makes a valve's switching threshold a property of the network state alone
# rather than of the pressure currently applied.
#
# The vector is ordered to match the node numbering above: [P_sink, P_source].
boundary_pressure_template = np.zeros(N_sink + N_source)
boundary_pressure_template[N_sink:] = 1

# ---------------------------------------------------------------------------
# Valve model
# ---------------------------------------------------------------------------
# When True, a valve responds to the magnitude of the pressure drop across it
# and not to its direction. This keeps every pressure drop in [0, 1]: without
# it, an edge that happens to be traversed "backwards" by the flow would
# report a negative drop, and its switching threshold would come out negative.
symmetric_flag = True

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------
# Each design inequality demands that one switching threshold be smaller than
# another. EPSILON is the margin by which it must be smaller, so the cost
# reaches zero only on a *strict* separation:
#
#     cost contribution = max(0, P_A - P_B + EPSILON)
#
# Without it the constraint would be the non-strict "P_A <= P_B", which two
# exactly equal thresholds would satisfy while leaving the network's behaviour
# genuinely ambiguous.
#
# The value is deliberately tiny. Its job is to rule out exact ties, not to
# impose a physical margin between thresholds. Switching thresholds here are
# of order 0.1-10, so this sits about ten orders of magnitude below the scale
# of the quantities being separated.
#
# Setting EPSILON = 0.0 recovers the non-strict form.
EPSILON = 1e-9
