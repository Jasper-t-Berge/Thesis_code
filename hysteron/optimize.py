"""
The CMA-ES search, in a form both experiments can share.

`cma_search` knows nothing about valves or topologies. It takes a cost
function and a way of drawing a starting point, and searches until the cost
reaches zero or the evaluation budget runs out. Everything problem-specific
stays with the caller.

Keeping it here rather than in either experiment folder is deliberate: the
benchmark and the topology sweep both run CMA-ES, and the two are meant to be
the same optimiser with the same settings. Sharing one implementation makes
that true by construction, instead of leaving two copies to drift apart.
"""

import cma
import numpy as np

# Population and parent counts. These are pycma's own defaults at 17
# dimensions -- 4 + floor(3 ln 17) = 12, and mu = lambda // 2 = 6 -- pinned
# explicitly so they stay put if the library's defaults ever change.
POPSIZE = 12    # lambda: candidates sampled per generation
MU      = 6     # mu: how many of those update the distribution

SIGMA0           = 0.3      # initial step size
SIGMA_FLOOR      = 0.01     # smallest step size allowed in any coordinate
STAGNATION_LIMIT = 100      # generations without improvement before a restart


def new_cma_es(x0, remaining_fevals, sigma0=SIGMA0, sigma_floor=SIGMA_FLOOR):
    """
    Create a CMA-ES instance centred on `x0`.

    `sigma_floor` is passed as pycma's `minstd`, which is the right place for
    it: the bound is then applied inside every step-size update rather than
    patched on afterwards. Without it, a flat plateau drives the step size
    towards zero, and once it collapses the search is trapped in a numerically
    tiny ball that even a restart cannot escape, because the restart mean is
    drawn from that same ball.

    pycma keeps its own random stream, and its `seed` option defaults to
    'time'. Seeding numpy alone therefore does *not* make a run reproducible.
    The seed is drawn here from numpy's already-seeded global generator, so a
    whole search -- including every restart, which needs its own stream or it
    would replay the same population -- follows from whatever seed the caller
    set.
    """
    opts = cma.CMAOptions()
    opts['maxfevals'] = remaining_fevals
    opts['verbose']   = -9
    opts['tolx']      = 1e-8
    opts['tolfun']    = 0
    opts['minstd']    = sigma_floor
    opts['popsize']   = POPSIZE
    opts['CMA_mu']    = MU
    opts['seed']      = int(np.random.randint(1, 2**31 - 1))
    return cma.CMAEvolutionStrategy(x0, sigma0, opts)


def cma_search(cost_fn, x0_fn, max_fevals,
               sigma0=SIGMA0, sigma_floor=SIGMA_FLOOR,
               stagnation_limit=STAGNATION_LIMIT,
               on_generation=None):
    """
    Run CMA-ES until the cost reaches zero or the budget is spent.

    Rather than nudging the parameters when progress stalls, as the
    gradient-based methods do, the whole search distribution is rebuilt. A
    stagnating CMA-ES usually has a covariance matrix that has grown into an
    unhelpful shape, and no amount of jostling the parameters will fix that.

    A restart re-centres on the best point found so far rather than on a fresh
    random one, and the global best cost is carried across it. Restarting from
    a random point would discard everything learned up to that moment, and
    resetting the best cost would blind the stagnation detector for the
    following `stagnation_limit` generations, since any finite cost looks like
    an improvement on infinity.

    Parameters
    ----------
    cost_fn          : callable taking a parameter vector, returning its cost
    x0_fn            : callable returning a random starting point
    max_fevals       : cost evaluations allowed
    sigma0           : initial step size
    sigma_floor      : minimum step size in any coordinate
    stagnation_limit : generations without improvement before a restart
    on_generation    : optional callback, called after each generation as
                       `on_generation(fevals, n_new, best_cost_overall)`

    Returns a dict with:
        success           whether the cost reached zero
        best_cost         lowest cost seen
        best_params       the parameter vector that achieved it
        cost_arr          running minimum, one entry per generation
        completion_index  generation the solution was found at, or -1
        num_restarts      how many times the distribution was rebuilt
        fevals            cost evaluations actually spent
    """
    best_params       = x0_fn()
    best_cost_overall = np.inf
    raw_cost_list     = [cost_fn(best_params)]
    gens_no_improve   = 0
    num_restarts      = 0
    fevals            = 0
    step              = 0
    completion_index  = -1

    es = new_cma_es(best_params, max_fevals, sigma0, sigma_floor)

    while fevals < max_fevals:

        # pycma's own convergence criteria fired, e.g. the distribution
        # collapsed onto a point. Restart rather than stop.
        if es.stop():
            es = new_cma_es(best_params, max_fevals - fevals, sigma0, sigma_floor)
            gens_no_improve = 0
            num_restarts   += 1

        solutions = es.ask()
        fitnesses = [cost_fn(s) for s in solutions]
        es.tell(solutions, fitnesses)

        best_cost = min(fitnesses)
        fevals   += len(solutions)
        step     += 1

        # Record a running minimum, so that len(cost_arr) * POPSIZE is a fair
        # evaluation count comparable with the other solvers.
        raw_cost_list.append(min(raw_cost_list[-1], best_cost))

        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_params       = solutions[int(np.argmin(fitnesses))].copy()
            gens_no_improve   = 0
        else:
            gens_no_improve  += 1

        if gens_no_improve >= stagnation_limit:
            es = new_cma_es(best_params, max_fevals - fevals, sigma0, sigma_floor)
            gens_no_improve = 0
            num_restarts   += 1

        if on_generation is not None:
            on_generation(fevals, len(solutions), best_cost_overall)

        if best_cost_overall == 0:
            completion_index = step
            break

    cost_arr = np.array(raw_cost_list)
    return {
        'success':          bool(cost_arr[-1] == 0),
        'best_cost':        float(cost_arr[-1]),
        'best_params':      np.array(best_params, dtype=float),
        'cost_arr':         cost_arr,
        'completion_index': completion_index,
        'num_restarts':     num_restarts,
        'fevals':           fevals,
    }
