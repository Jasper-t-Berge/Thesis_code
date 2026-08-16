"""
Bayesian optimisation solver, TPE variant (Tree-structured Parzen Estimator).

Written from scratch on numpy alone; no optuna or scikit-learn needed.

A Bayesian optimiser avoids expensive cost evaluations by building a cheap
surrogate model of the landscape and consulting that instead, only evaluating
the real cost at the point the surrogate thinks is most promising.

Why TPE rather than the more familiar Gaussian process with expected
improvement? A GP with an RBF kernel assumes a smooth landscape, and this one
is not: the cost is piecewise linear with a large flat zero basin, so the
kernel can represent neither the cliffs nor the flat regions and the
acquisition function comes out systematically wrong. TPE makes no smoothness
assumption at all.

How it works: after an initial random phase, split every observation so far at
a quantile of the cost into a "good" set and a "bad" set, fit a kernel density
estimate to each, then sample candidates from the good density and score them
by the ratio l(x)/g(x). A high ratio means a candidate looks like the points
that worked and unlike the points that did not. The best-scoring candidate is
evaluated next, and the split is recomputed with it included.

Hyperparameters: 30 random startup evaluations, the best 15% counted as
"good", 64 candidates scored per proposal, and a kick whenever 200 consecutive
evaluations fail to improve the cost by 1%.

TPE gets more expensive with every iteration, because its surrogate uses every
past evaluation: the per-iteration surrogate cost scales as the number of
observations times the number of candidates. That overhead is why this method
is worth judging on wall-clock time and not only on evaluation count.
"""

import pathlib
import sys
import warnings

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import numpy as np                                            # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

import problem                                                # noqa: E402
from hysteron import (                                        # noqa: E402
    HysteronValve,
    cost_function_system,
    give_kick,
    set_RNG_seed,
    straighten_parameters,
)

N_PARAMS = problem.N_params
N_VALVES = problem.N_valves

# --------------------------------------------------------------------------
# Physical bounds of the search box
# Layout: [h- x4, h+ x4, C0 x4, dC x4, conds x1]
# --------------------------------------------------------------------------
_LO = np.array(
    [problem.H_LOWER_BOUND] * N_VALVES +          # h-
    [problem.H_LOWER_BOUND] * N_VALVES +          # h+
    [problem.C_LOWER_BOUND] * N_VALVES +          # C0
    [-4.5]                  * N_VALVES +          # dC, always negative
    [problem.C_LOWER_BOUND] * problem.N_conds     # conductors
)
_HI = np.array(
    [problem.H_UPPER_BOUND] * N_VALVES +
    [problem.H_UPPER_BOUND] * N_VALVES +
    [problem.C_UPPER_BOUND] * N_VALVES +
    [-0.1]                  * N_VALVES +
    [problem.C_UPPER_BOUND] * problem.N_conds
)


def _safe_cost(valves, conds):
    """
    Evaluate the cost, mapping any numerical failure to a large finite value.

    TPE has to be handed a number for every candidate it proposes; a NaN or an
    exception from a near-singular network would poison the density estimates,
    so such candidates are simply scored as very bad.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            c = cost_function_system(
                problem.ineq_array, valves, conds,
                problem.boundary_pressure_template, problem.incidence_matrix)
            return float(c) if np.isfinite(c) else 1e6
        except Exception:
            return 1e6


def _evaluate(params):
    """Straighten a raw proposal, build the objects, and score it."""
    valves = HysteronValve(
        np.zeros(N_VALVES),
        params[:N_VALVES].copy(),
        params[N_VALVES:2 * N_VALVES].copy(),
        params[2 * N_VALVES:3 * N_VALVES].copy(),
        params[3 * N_VALVES:4 * N_VALVES].copy(),
    )
    conds = params[4 * N_VALVES:].copy()
    params, valves, conds = straighten_parameters(params.copy(), valves, conds)
    cost = _safe_cost(valves, conds)
    return cost, valves, conds, params


def _normalise(params):
    """Physical parameters -> the unit box TPE works in."""
    return (params - _LO) / (_HI - _LO)


def _denormalise(x):
    """Unit box -> physical parameters."""
    return _LO + x * (_HI - _LO)


class TPE:
    """
    A minimal Tree-structured Parzen Estimator.

    Keeps every observation, and on request splits them at the `gamma`
    quantile of the cost into good and bad sets, builds a product of
    one-dimensional Gaussian kernel density estimates over each, and returns
    the candidate that maximises log l(x) - log g(x).

    Treating the dimensions independently is what the "tree-structured" name
    refers to; it is also what keeps the density estimate tractable in 17
    dimensions with only a few hundred observations, where a full joint
    density would be hopelessly under-determined.

    Parameters
    ----------
    n_startup    : purely random evaluations before the model switches on
    gamma        : quantile cut; the lowest-cost `gamma` fraction is "good"
    bandwidth    : kernel width, as a multiplier on Scott's rule
    n_candidates : candidates scored per proposal
    """

    def __init__(self, n_startup=30, gamma=0.15, bandwidth=0.3, n_candidates=64):
        self.n_startup    = n_startup
        self.gamma        = gamma
        self.bw           = bandwidth
        self.n_candidates = n_candidates
        self.X = []   # normalised parameter vectors
        self.y = []   # their costs

    def observe(self, x_norm, cost):
        """Record one evaluated point."""
        self.X.append(x_norm.copy())
        self.y.append(cost)

    def suggest(self):
        """Propose the next point to evaluate, in normalised coordinates."""
        n = len(self.y)

        # Random exploration phase: with too few observations the split would
        # be meaningless.
        if n < self.n_startup:
            return np.random.uniform(0, 1, N_PARAMS)

        X = np.array(self.X)
        y = np.array(self.y)

        threshold = np.quantile(y, self.gamma)
        good_mask = y <= threshold

        # A density estimate needs at least a couple of points per side. On a
        # plateau the quantile can put everything on one side, so fall back to
        # a random draw rather than build a degenerate model.
        if good_mask.sum() < 2 or (~good_mask).sum() < 2:
            return np.random.uniform(0, 1, N_PARAMS)

        X_good = X[good_mask]
        X_bad  = X[~good_mask]

        candidates = self._kde_sample(X_good, self.n_candidates)
        candidates = np.clip(candidates, 0.0, 1.0)

        log_l = self._kde_logpdf(X_good, candidates)
        log_g = self._kde_logpdf(X_bad,  candidates)
        score = log_l - log_g          # the log of l(x)/g(x); maximise

        return candidates[np.argmax(score)]

    def _bandwidth(self, X_group):
        """Scott's rule for this group, scaled by the configured factor."""
        n, d = X_group.shape
        return self.bw * n ** (-1.0 / (d + 4))

    def _kde_sample(self, X_group, n_samples):
        """Sample the density: pick an observed point, then jitter every axis."""
        h     = self._bandwidth(X_group)
        idx   = np.random.randint(0, len(X_group), n_samples)
        noise = np.random.normal(0, h, (n_samples, N_PARAMS))
        return X_group[idx] + noise

    def _kde_logpdf(self, X_group, candidates):
        """
        Log density at each candidate.

        Computed in log space with the largest term factored out before
        exponentiating, because a product of 17 per-dimension kernels
        underflows a float otherwise.
        """
        h = self._bandwidth(X_group)
        diff = candidates[:, None, :] - X_group[None, :, :]
        log_k = -0.5 * (diff / h) ** 2 - np.log(h * np.sqrt(2 * np.pi))
        log_k_prod = log_k.sum(axis=2)                     # product over dims
        log_k_max  = log_k_prod.max(axis=1, keepdims=True)
        log_pdf    = log_k_max.squeeze() + np.log(
            np.exp(log_k_prod - log_k_max).sum(axis=1)) - np.log(len(X_group))
        return log_pdf


def run(seed, num_steps=problem.EVAL_BUDGET, num_sim=1,
        n_startup=30, gamma=0.15, bandwidth=0.3, n_candidates=64,
        kick_interval=200, kick_fraction=0.01,
        data_filename=None, show_plot=True):
    """
    Run the TPE solver.

    Every step is exactly one cost evaluation, so `num_steps` is the budget
    directly.

    Returns a list of result dicts, one per simulation, with keys:
        success, final_cost, completion_index, num_kick, cost_arr, params

    `params` is the best parameter vector found, which is the actual answer
    to the design problem; the rest describe how the search got there.
    """
    set_RNG_seed(seed)

    all_results = []

    for _sim in range(num_sim):

        raw_cost         = np.full(num_steps, np.inf)
        completion_index = -1
        num_kick         = 0
        y_best           = np.inf
        best_params      = None
        cur_valves       = None
        cur_conds        = None
        cost_at_window_start = np.inf

        tpe = TPE(n_startup=n_startup, gamma=gamma,
                  bandwidth=bandwidth, n_candidates=n_candidates)

        for step in range(num_steps):

            # Stagnation check over the last window of evaluations.
            if step > 0 and step % kick_interval == 0 and best_params is not None:
                window_min  = np.min(raw_cost[max(0, step - kick_interval):step])
                improvement = ((cost_at_window_start - window_min)
                               / max(abs(cost_at_window_start), 1e-12))
                cost_at_window_start = window_min

                if improvement < kick_fraction:
                    num_kick += 1
                    _, _, kicked_p = give_kick(
                        best_params.copy(), cur_valves, cur_conds)
                    cost, v, c, p = _evaluate(kicked_p)
                    raw_cost[step] = cost
                    tpe.observe(_normalise(p), cost)
                    if cost < y_best:
                        y_best, best_params, cur_valves, cur_conds = cost, p.copy(), v, c
                    if cost == 0:
                        completion_index = step
                        break
                    continue

            x_norm = tpe.suggest()
            params = _denormalise(x_norm)
            cost, v, c, p = _evaluate(params)
            raw_cost[step] = cost

            # Record the straightened parameters rather than the raw proposal,
            # so the model learns the feasible space it is actually searching.
            tpe.observe(_normalise(p), cost)

            if cost < y_best:
                y_best      = cost
                best_params = p.copy()
                cur_valves  = v
                cur_conds   = c
                if step == 0:
                    cost_at_window_start = cost

            if cost == 0:
                completion_index = step
                break

        # TPE jumps around, so the raw trace is not monotonic. Convert it to a
        # running minimum, which is what the other solvers report.
        final_step = step + 1
        cost_arr   = raw_cost[:final_step].copy()
        for s in range(1, final_step):
            if cost_arr[s] > cost_arr[s - 1]:
                cost_arr[s] = cost_arr[s - 1]

        if data_filename:
            with open(data_filename, "a") as f:
                f.write(f"{cost_arr[0]},{cost_arr[-1]},{completion_index},{num_kick}\n")

        all_results.append({
            'success':          cost_arr[-1] == 0,
            'final_cost':       float(cost_arr[-1]),
            'completion_index': completion_index,
            'num_kick':         num_kick,
            'cost_arr':         cost_arr,
            'params':           None if best_params is None else best_params.copy(),
        })

    if show_plot:
        plt.figure()
        plt.title("Bayesian optimisation (TPE) - best cost over evaluations")
        plt.xlabel("Cost function evaluation")
        plt.ylabel("Best cost so far")
        plt.plot(all_results[-1]['cost_arr'])
        plt.yscale('log')
        plt.tight_layout()
        plt.show(block=True)

    return all_results


if __name__ == "__main__":
    results = run(seed=-1, num_sim=1, show_plot=True)
    r = results[0]
    print(f"Success:           {r['success']}")
    print(f"Final cost:        {r['final_cost']:.6f}")
    print(f"Completion index:  {r['completion_index']}")
    print(f"Kicks applied:     {r['num_kick']}")
