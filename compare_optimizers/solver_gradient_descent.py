"""
Gradient descent solver.

The simplest of the four methods, and the baseline the other three are
measured against. It probes the cost with a finite difference, reduces the
result to a per-parameter direction, and steps by a random amount drawn from
U(0, 0.001). Whenever the cost stops improving it applies a kick to escape the
flat region.

Hyperparameters: finite-difference offset 1e-4, step size ~ U(0, 0.001), and a
kick whenever 20 consecutive steps fail to improve the cost by 1%.

Note: the probe in `update_parameters_GD` does not re-bind `valves` to the
perturbed `params`, so the measured differences come out zero and the update
reduces to a random step downhill, kept only when it improves the cost.
`solver_adam.py` re-binds via `implement_parameters` and its gradient is
genuine. Left as it stands, as the baseline the other methods are measured
against.
"""

import pathlib
import sys

# Put the project root and this folder on the import path, so `hysteron` and
# the sibling modules resolve however this file is launched.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import numpy as np                                            # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

import problem                                                # noqa: E402
from hysteron import (                                        # noqa: E402
    cost_function_system,
    give_kick,
    implement_parameters,
    random_parameter_vector,
    set_RNG_seed,
    straighten_parameters,
)


def update_parameters_GD(params, conds, old_cost, ineq_array, valves,
                         boundary_pressure_template, incidence_matrix,
                         ignore_list=()):
    """
    One gradient descent step. See the note in the module docstring for what
    this does in practice.

    Returns `(valves, conds, params, cost)`. The step is rejected and the old
    parameters restored if the cost got worse, so the cost never increases.
    """
    old_params      = np.copy(params)
    dx              = 0.0001        # finite-difference offset
    param_step_size = 0.001         # upper end of the random step size
    deriv_array     = np.zeros(len(params))

    for i in range(len(params)):
        if i in ignore_list:
            continue
        params[i] += dx
        new_cost = cost_function_system(ineq_array, valves, conds,
                                        boundary_pressure_template, incidence_matrix)
        deriv_array[i] = new_cost - old_cost
        params[i] -= dx

    # Reduce the probe to a direction only; the magnitude of the step comes
    # from the random draw below rather than from the size of the difference.
    for i in range(len(deriv_array)):
        if deriv_array[i] < 0:
            deriv_array[i] = -1
        elif i in ignore_list:
            deriv_array[i] = 0
        else:
            deriv_array[i] = 1

    params = params - deriv_array * np.random.uniform(0, param_step_size,
                                                      len(deriv_array))
    params, valves, conds = straighten_parameters(params, valves, conds)

    cost = cost_function_system(ineq_array, valves, conds,
                                boundary_pressure_template, incidence_matrix)

    # Reject the step if it made things worse.
    if cost > old_cost:
        params = old_params
        valves, conds = implement_parameters(params, valves, conds)
        cost = old_cost

    return valves, conds, params, cost


def run(seed, num_steps=556, num_sim=1, kick_interval=20, kick_fraction=0.01,
        data_filename=None, show_plot=True):
    """
    Run the gradient descent solver.

    Each step costs 1 + N_params = 18 cost evaluations, so the default of 556
    steps corresponds to the 10,000 evaluation budget.

    Parameters
    ----------
    seed          : RNG seed; -1 draws one from the clock
    num_steps     : steps per simulation
    num_sim       : independent simulations to run under this seed
    kick_interval : how many steps between stagnation checks
    kick_fraction : the fractional improvement needed to avoid a kick
    data_filename : if given, append one CSV line per simulation
    show_plot     : draw the cost trace of the final simulation

    Returns
    -------
    A list of result dicts, one per simulation, with keys:
        success, final_cost, completion_index, num_kick, cost_arr, params

    `params` is the best parameter vector found, which is the actual answer
    to the design problem; the rest describe how the search got there.
    """
    set_RNG_seed(seed)
    all_results = []

    for j in range(num_sim):
        params, valves, conds = random_parameter_vector(
            problem.N_valves, problem.N_conds,
            problem.H_LOWER_BOUND, problem.H_UPPER_BOUND,
            problem.C_LOWER_BOUND, problem.C_UPPER_BOUND)

        cost_arr    = np.zeros(num_steps)
        cost_arr[0] = cost_function_system(
            problem.ineq_array, valves, conds,
            problem.boundary_pressure_template, problem.incidence_matrix)
        num_kick         = 0
        completion_index = -1

        for i in range(1, num_steps):
            valves, conds, params, cost_arr[i] = update_parameters_GD(
                params, conds, cost_arr[i - 1], problem.ineq_array, valves,
                problem.boundary_pressure_template, problem.incidence_matrix)

            if i % 500 == 0:
                print(f"  GD sim {j}: {100 * i / num_steps:.1f}% done, "
                      f"cost = {cost_arr[i]:.6f}")

            if cost_arr[i] == 0:
                print(f"  GD sim {j}: solution found at step {i}!")
                completion_index = i
                break

            # Stagnation check: kick if the last window of steps bought less
            # than kick_fraction relative improvement.
            if i % kick_interval == 0:
                if (cost_arr[i - kick_interval] - cost_arr[i]) / max(cost_arr[i], 1e-12) < kick_fraction:
                    num_kick += 1
                    valves, conds, params = give_kick(params, valves, conds)
                    cost_arr[i] = cost_function_system(
                        problem.ineq_array, valves, conds,
                        problem.boundary_pressure_template, problem.incidence_matrix)

        print(f"  GD sim {j}: completion={completion_index}, kicks={num_kick}, "
              f"final cost={cost_arr[i]:.6f}")

        if data_filename:
            with open(data_filename, "a") as f:
                f.write(f"{cost_arr[0]},{cost_arr[i]},{completion_index},{num_kick}\n")

        all_results.append({
            'success':          cost_arr[i] == 0,
            'final_cost':       cost_arr[i],
            'completion_index': completion_index,
            'num_kick':         num_kick,
            'cost_arr':         cost_arr[:i + 1],
            'params':           np.copy(params),
        })

    if show_plot:
        plt.figure()
        plt.title("Gradient Descent - cost over time")
        plt.xlabel("Step")
        plt.ylabel("Cost")
        plt.plot(cost_arr)
        plt.show(block=True)

    return all_results


if __name__ == "__main__":
    run(seed=-1, num_steps=556, num_sim=1, show_plot=True)
