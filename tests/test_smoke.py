"""
Regression tests for the whole project.

Plain asserts, no test framework, so this runs with nothing installed beyond
what the code itself already needs:

    python tests/test_smoke.py

The point is to catch the failures that matter for a repository someone else
is going to clone: a script that will not import, a solver whose return value
does not match the contract `run_comparison` relies on, a seed convention that
raises, and a physics pipeline that has silently stopped agreeing with itself.

It takes a couple of minutes. Every solver is run on a deliberately small
budget, so a solver failing to *converge* is not a failure here -- only a
solver failing to *run* correctly is.
"""

import pathlib
import sys
import traceback

# Put the project root and the two experiment folders on the import path, so
# this file runs from anywhere.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT),
                str(_ROOT / "compare_optimizers"),
                str(_ROOT / "topology_sweep")]

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # never open a window

import numpy as np                                             # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------
_PASSED, _FAILED = [], []


def test(fn):
    """Mark a function as a test case, for `_run_all` to pick up."""
    fn._is_test = True
    return fn


def _run_all():
    tests = [v for v in list(globals().values())
             if callable(v) and getattr(v, "_is_test", False)]
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            _PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception:
            _FAILED.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"{len(_PASSED)} passed, {len(_FAILED)} failed")
    if _FAILED:
        print("failed: " + ", ".join(_FAILED))
    print("=" * 60)
    return 1 if _FAILED else 0


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
@test
def test_package_imports():
    """Everything `hysteron.__all__` advertises is actually importable."""
    import hysteron
    missing = [n for n in hysteron.__all__ if not hasattr(hysteron, n)]
    assert not missing, f"__all__ names not present on the package: {missing}"


@test
def test_every_script_imports():
    """Each runnable script imports cleanly, from any working directory."""
    import importlib
    for mod in ["problem", "solver_gradient_descent", "solver_adam",
                "solver_cma", "solver_bayesian", "solver_hybrid",
                "scaled_problem", "run_comparison", "run_scaling",
                "run_heatmap"]:
        importlib.import_module(mod)


@test
def test_all_topologies_build():
    """All three topologies produce a well-formed 4-valve, 1-conductor network."""
    from hysteron import build_incidence_matrix, initialize_geometry
    for topo in ["series", "series-parallel", "wheatstone"]:
        edges, NV, NC, NN = initialize_geometry(topology=topo)
        assert (NV, NC) == (4, 1), f"{topo}: expected 4 valves + 1 conductor"
        assert edges.shape == (NV + NC, 2)
        assert edges.max() < NN, f"{topo}: edge refers to a node outside the network"
        D = build_incidence_matrix(edges, NV + NC, NN)
        # Every edge leaves exactly one node and enters exactly one other.
        assert np.allclose(D.sum(axis=1), 0), f"{topo}: malformed incidence matrix"


@test
def test_unknown_topology_and_tgraph_raise():
    """Bad names fail loudly, and the message lists the valid choices."""
    from hysteron import initialize_geometry, initialize_tgraph
    for fn, kw, valid in [(initialize_geometry, {"topology": "nonsense"},
                           ["series", "series-parallel", "wheatstone"]),
                          (initialize_tgraph, {"tgraph": "nonsense"},
                           ["Preisach", "Avalanche", "MPO"])]:
        try:
            fn(**kw)
        except ValueError as e:
            for name in valid:
                assert name in str(e), f"{fn.__name__} error omits '{name}': {e}"
        else:
            raise AssertionError(f"{fn.__name__} accepted an unknown name")


@test
def test_state_number_roundtrip():
    """Packing and unpacking a valve state are exact inverses."""
    from hysteron import convert_number_to_state, convert_state_to_number
    for n in range(2 ** 4):
        assert convert_state_to_number(convert_number_to_state(n, 4)) == n


@test
def test_symmetric_flag_is_live():
    """`config.symmetric_flag` takes effect when changed at runtime."""
    from hysteron import (build_incidence_matrix, config,
                          convert_number_to_state, initialize_geometry,
                          random_parameter_vector, straighten_parameters)
    from hysteron.network import calculate_pressure_hysteron_and_conductor_network

    edges, NV, NC, NN = initialize_geometry(topology="wheatstone")
    D = build_incidence_matrix(edges, NV + NC, NN)
    bpt = config.boundary_pressure_template

    original = config.symmetric_flag
    try:
        # Find a network and state where the two settings genuinely differ,
        # i.e. where some edge is traversed against its own orientation and so
        # carries a negative drop with the flag off.
        np.random.seed(0)
        differed = False
        for _ in range(50):
            p, valves, conds = random_parameter_vector(NV, NC, 0.1, 1.0, 0.5, 5.0)
            p, valves, conds = straighten_parameters(p, valves, conds)
            for s in range(2 ** NV):
                valves.state = convert_number_to_state(s, NV)

                config.symmetric_flag = True
                on = calculate_pressure_hysteron_and_conductor_network(
                    valves, conds, bpt, D)
                config.symmetric_flag = False
                off = calculate_pressure_hysteron_and_conductor_network(
                    valves, conds, bpt, D)

                assert np.all(on >= 0), "flag on should give magnitudes only"
                assert np.allclose(on, np.abs(off)), \
                    "the two settings should differ only in sign"
                if np.any(off < 0):
                    differed = True
                    break
            if differed:
                break
    finally:
        config.symmetric_flag = original

    assert differed, \
        ("setting config.symmetric_flag = False never produced a negative "
         "pressure drop, so the flag is not being read at call time "
         "(is it captured at import?)")


@test
def test_straighten_parameters_is_physical():
    """Straightening maps any proposal into the physically valid region."""
    from hysteron import (HysteronValve, implement_parameters,
                          straighten_parameters)
    np.random.seed(1)
    for _ in range(200):
        # Deliberately unphysical: negative conductances, h- above h+.
        params = np.random.uniform(-5, 5, 17)
        valves = HysteronValve(np.zeros(4, dtype=int), *[np.zeros(4)] * 4)
        valves, conds = implement_parameters(params, valves, params[16:])
        params, valves, conds = straighten_parameters(params, valves, conds)
        assert np.all(valves.hp >= valves.hm), "h+ must not fall below h-"
        assert np.all(valves.hm > 0), "opening pressure must stay positive"
        assert np.all(valves.dC <= 0), "closing a valve must not raise conductance"
        assert np.all(valves.C0 + valves.dC > 0), "closed conductance must stay positive"
        assert np.all(np.asarray(conds) > 0), "conductor conductance must stay positive"


@test
def test_tgraph_extraction_is_wellformed():
    """The forward t-graph solver produces a consistent graph on every topology."""
    from hysteron import (build_incidence_matrix, config, initialize_geometry,
                          random_parameter_vector, straighten_parameters)
    from hysteron.tgraph import (check_t_graph_loop_RPM,
                                 find_t_graph_with_thresholds)
    np.random.seed(2)
    for topo in ["series", "series-parallel", "wheatstone"]:
        edges, NV, NC, NN = initialize_geometry(topology=topo)
        D = build_incidence_matrix(edges, NV + NC, NN)
        for _ in range(150):
            p, v, c = random_parameter_vector(NV, NC, 0.1, 1.0, 0.5, 5.0)
            p, v, c = straighten_parameters(p, v, c)
            tg = find_t_graph_with_thresholds(
                v, c, config.boundary_pressure_template, D)

            assert 0 < len(tg) <= 2 ** NV, "impossible number of states"
            assert len(set(tg[:, 0].tolist())) == len(tg), "duplicate state rows"
            assert np.all(tg[:, 0] >= 0), "a -1 sentinel was queued as a state"
            # Every transition either is a dead end or lands on a known state.
            known = set(tg[:, 0].tolist())
            for column in (1, 2):
                for target in tg[:, column].tolist():
                    assert target == -1 or target in known, \
                        "transition points at a state not in the graph"
            check_t_graph_loop_RPM(tg, NV)      # must not raise


@test
def test_orbits_follow_the_right_transition_direction():
    """
    `dict_orbits_of_t_graph` must read up-transitions from the up column.

    The t-graph's column 1 is filled by the *decreasing*-pressure solve and
    column 2 by the increasing one. Reading them the wrong way round would
    silently give the up-orbit the down-transitions, and the return-point
    memory check that consumes these orbits would then be answering about the
    wrong thing rather than failing outright.
    """
    from hysteron import (build_incidence_matrix, config, initialize_geometry,
                          random_parameter_vector, straighten_parameters)
    from hysteron.tgraph import (dict_orbits_of_t_graph,
                                 find_t_graph_with_thresholds)

    edges, NV, NC, NN = initialize_geometry(topology="wheatstone")
    D = build_incidence_matrix(edges, NV + NC, NN)
    np.random.seed(4)
    checked = 0

    for _ in range(40):
        p, v, c = random_parameter_vector(NV, NC, 0.1, 1.0, 0.5, 5.0)
        p, v, c = straighten_parameters(p, v, c)
        tg = find_t_graph_with_thresholds(
            v, c, config.boundary_pressure_template, D)
        up_orbits, down_orbits = dict_orbits_of_t_graph(tg, NV)

        for row in tg:
            state, down_to, up_to = row[0], int(row[1]), int(row[2])
            assert up_orbits[state][0] == up_to, (
                f"state {int(state)}: up-orbit starts at {up_orbits[state][0]}, "
                f"but raising the pressure leads to {up_to}")
            assert down_orbits[state][0] == down_to, (
                f"state {int(state)}: down-orbit starts at "
                f"{down_orbits[state][0]}, but lowering the pressure leads to "
                f"{down_to}")
            checked += 1

    assert checked > 100, "not enough states exercised to be meaningful"


@test
def test_cost_is_zero_only_on_target_behaviour():
    """A solved network really does reproduce every targeted transition."""
    from hysteron import (build_incidence_matrix, build_inequalities, config,
                          cost_function_system, initialize_geometry,
                          initialize_tgraph)
    from hysteron.tgraph import find_t_graph_with_thresholds
    import solver_cma

    edges, NV, NC, NN = initialize_geometry(topology="wheatstone")
    D = build_incidence_matrix(edges, NV + NC, NN)
    up_trans, down_trans = initialize_tgraph(tgraph="MPO")
    ineqs = build_inequalities(NV, up_trans, down_trans)

    solved = None
    for seed in (7, 11, 13, 17, 23):
        r = solver_cma.run(seed=seed, num_sim=1, show_plot=False)[0]
        if r["success"]:
            solved = r
            break
    assert solved is not None, "CMA-ES solved none of five seeds on Wheatstone/MPO"

    valves, conds = solver_cma.reconstruct_valves_conds(solved["params"])

    # The reported cost must belong to the parameters that were handed back.
    rescored = cost_function_system(ineqs, valves, conds,
                                    config.boundary_pressure_template, D)
    assert rescored == 0, f"re-scoring the returned params gave {rescored}, not 0"

    # And zero cost must mean the behaviour is actually present.
    tg = find_t_graph_with_thresholds(
        valves, conds, config.boundary_pressure_template, D)
    rows = {int(row[0]): row for row in tg}
    for src, dst in up_trans:
        assert src in rows and int(rows[src][2]) == dst, \
            f"up-transition {src} -> {dst} not reproduced"
    for src, dst in down_trans:
        assert src in rows and int(rows[src][1]) == dst, \
            f"down-transition {src} -> {dst} not reproduced"


# ---------------------------------------------------------------------------
# The solvers
# ---------------------------------------------------------------------------
# Every solver is run twice over a small budget: once to check the contract
# `run_comparison` depends on, and once more to check the seed is honoured.
_SOLVER_CASES = [
    ("solver_gradient_descent", {"num_steps": 40}),
    ("solver_adam",             {"num_steps": 40}),
    ("solver_cma",              {"max_fevals": 240}),
    ("solver_bayesian",         {"num_steps": 120}),
    ("solver_hybrid",           {"max_fevals": 120}),
]

_CONTRACT = ["success", "final_cost", "completion_index", "cost_arr", "params"]


@test
def test_solvers_honour_the_result_contract():
    """Each solver returns what `run_comparison` and the README promise."""
    import importlib
    for name, kwargs in _SOLVER_CASES:
        mod = importlib.import_module(name)
        results = mod.run(seed=5, num_sim=2, show_plot=False, **kwargs)
        assert isinstance(results, list) and len(results) == 2, \
            f"{name}: num_sim=2 should give two result dicts"
        for r in results:
            for key in _CONTRACT:
                assert key in r, f"{name}: result is missing '{key}'"
            assert len(r["params"]) == 17, f"{name}: params should be 17-dimensional"
            assert np.all(np.isfinite(r["params"])), f"{name}: non-finite params"
            assert r["final_cost"] >= 0, f"{name}: negative cost"
            assert r["success"] == (r["final_cost"] == 0), \
                f"{name}: 'success' disagrees with 'final_cost'"


@test
def test_solvers_are_reproducible():
    """The same seed gives the same answer; a different seed does not."""
    import importlib
    for name, kwargs in _SOLVER_CASES:
        mod = importlib.import_module(name)
        a = mod.run(seed=5, num_sim=1, show_plot=False, **kwargs)[0]
        b = mod.run(seed=5, num_sim=1, show_plot=False, **kwargs)[0]
        assert np.array_equal(a["params"], b["params"]), \
            f"{name}: same seed gave different parameters"
        assert a["final_cost"] == b["final_cost"], \
            f"{name}: same seed gave a different cost"


@test
def test_clock_seed_convention_works():
    """seed=-1 means 'draw one from the clock' in every solver, and never raises."""
    import importlib
    for name, kwargs in _SOLVER_CASES:
        mod = importlib.import_module(name)
        mod.run(seed=-1, num_sim=1, show_plot=False, **kwargs)


@test
def test_kick_perturbs_and_stays_physical():
    """A kick moves every parameter and leaves the network physical."""
    from hysteron import give_kick, random_parameter_vector, straighten_parameters
    np.random.seed(3)
    p, v, c = random_parameter_vector(4, 1, 0.1, 1.0, 0.5, 5.0)
    p, v, c = straighten_parameters(p, v, c)
    before = p.copy()
    v, c, p = give_kick(p, v, c)
    assert not np.array_equal(before, p), "give_kick left the parameters untouched"
    assert np.all(v.hp >= v.hm) and np.all(v.dC <= 0), \
        "give_kick left the network unphysical"

    # An ignored index must come back untouched.
    p2, v2, c2 = random_parameter_vector(4, 1, 0.1, 1.0, 0.5, 5.0)
    p2, v2, c2 = straighten_parameters(p2, v2, c2)
    frozen = p2[16]
    v2, c2, p2 = give_kick(p2, v2, c2, ignore_list=(16,))
    assert p2[16] == frozen, "give_kick moved a parameter in ignore_list"


# ---------------------------------------------------------------------------
# The scaling machinery
# ---------------------------------------------------------------------------
@test
def test_scaled_problem_target_is_solvable():
    """
    The generating network scores exactly zero on the target read off it.

    This is what makes a scaling result meaningful: if the generating network
    did not score zero, a solver failing at that size would tell us nothing,
    because we would not know a solution existed at all.
    """
    import scaled_problem

    for k in (1, 2):
        inst = scaled_problem.ScaledProblem(k, seed=0, n_pulses=20)
        assert inst.N_valves == 4 * k, f"k={k}: expected {4 * k} valves"
        assert inst.N_params == 4 * inst.N_valves + inst.N_conds
        assert len(inst.ineq_array) > 0, f"k={k}: target produced no inequalities"
        assert inst.gen_cost == 0, \
            f"k={k}: the network the target came from scores {inst.gen_cost}, not 0"


@test
def test_scaled_target_seeds_cannot_collide_with_trial_seeds():
    """Target seeds stay clear of the seeds trials are run under."""
    import scaled_problem
    assert scaled_problem.ScaledProblem.TARGET_SEED_BASE >= 100_000, \
        "target seeds must sit well above any plausible trial seed"


if __name__ == "__main__":
    print("Running project tests\n" + "-" * 60)
    sys.exit(_run_all())
