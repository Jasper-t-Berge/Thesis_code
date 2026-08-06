"""
Adam solver.

Takes the same finite-difference gradient as plain gradient descent, but
instead of discarding the magnitude and stepping randomly, it feeds the
gradient into Adam's moment update so that each parameter gets its own
adaptive step size.

The two moments are running averages of the gradient and of its square. The
first gives the step a persistent direction, so progress accumulates along
consistent slopes; the second measures how volatile each parameter's gradient
has been, and dividing by it damps the parameters that keep changing their
mind. Both start at zero, which would make the first few steps far too small,
so they are bias-corrected by the `1 - beta**t` factors below.

Hyperparameters: learning rate 0.01, decay rates 0.9 and 0.999, a 1e-8 guard
against division by zero, and a kick whenever 20 consecutive steps fail to
improve the cost by 1%.
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


def update_parameters_Adam(params, conds, ineq_array, valves,
                           boundary_pressure_template, incidence_matrix,
                           m, v, t,
                           lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                           ignore_list=()):
    """
    One Adam step, using a numerical finite-difference gradient.

    Note the `implement_parameters` calls around each probe: they re-bind
    `valves` to the perturbed `params`, which is what makes the perturbation
    visible to the cost function and the gradient genuine.

    Parameters
    ----------
    m, v        : the first and second moment vectors, owned by the caller
    t           : Adam time step, starting at 1, incremented by the caller
    lr          : learning rate
    beta1/beta2 : moment decay rates
    eps         : guard against division by zero
    ignore_list : parameter indices to freeze

    Returns `(valves, conds, params, m, v)`.
    """
    dx = 0.0001

    old_cost = cost_function_system(ineq_array, valves, conds,
                                    boundary_pressure_template, incidence_matrix)
    grad = np.zeros(len(params))

    for i in range(len(params)):
        if i in ignore_list:
            continue
        params[i] += dx
        valves, conds = implement_parameters(params, valves, conds)
        new_cost = cost_function_system(ineq_array, valves, conds,
                                        boundary_pressure_template, incidence_matrix)
        grad[i] = (new_cost - old_cost) / dx
        params[i] -= dx
        valves, conds = implement_parameters(params, valves, conds)

    # Running averages of the gradient and of its square.
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2

    # Bias correction, undoing the fact that both moments started at zero.
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Per-parameter step: direction from the first moment, scale damped by the
    # volatility recorded in the second.
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    params, valves, conds = straighten_parameters(params, valves, conds)

    return valves, conds, params, m, v


def run(seed, num_steps=556, num_sim=1, lr=0.01, kick_interval=20,
        kick_fraction=0.01, data_filename=None, show_plot=True):
    """
    Run the Adam solver.

    Each step costs 1 + N_params = 18 evaluations for the gradient plus one to
    record the new cost, so 556 steps sits at roughly the 10,000 evaluation
    budget.

    Returns a list of result dicts, one per simulation, with keys:
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

        # Adam state, reset for each simulation.
        m_adam = np.zeros(problem.N_params)
        v_adam = np.zeros(problem.N_params)
        t_adam = 1

        cost_arr    = np.zeros(num_steps)
        cost_arr[0] = cost_function_system(
            problem.ineq_array, valves, conds,
            problem.boundary_pressure_template, problem.incidence_matrix)
        num_kick         = 0
        completion_index = -1

        for i in range(1, num_steps):
            valves, conds, params, m_adam, v_adam = update_parameters_Adam(
                params, conds, problem.ineq_array, valves,
                problem.boundary_pressure_template, problem.incidence_matrix,
                m_adam, v_adam, t_adam, lr=lr)
            t_adam += 1
            cost_arr[i] = cost_function_system(
                problem.ineq_array, valves, conds,
                problem.boundary_pressure_template, problem.incidence_matrix)

            if i % 500 == 0:
                print(f"  Adam sim {j}: {100 * i / num_steps:.1f}% done, "
                      f"cost = {cost_arr[i]:.6f}")

            if cost_arr[i] == 0:
                print(f"  Adam sim {j}: solution found at step {i}!")
                completion_index = i
                break

            if i % kick_interval == 0:
                if (cost_arr[i - kick_interval] - cost_arr[i]) / max(cost_arr[i], 1e-12) < kick_fraction:
                    num_kick += 1
                    valves, conds, params = give_kick(params, valves, conds)
                    # The moments average over gradients from the basin we
                    # have just left, so they are reset along with the clock
                    # that bias-corrects them.
                    m_adam = np.zeros(problem.N_params)
                    v_adam = np.zeros(problem.N_params)
                    t_adam = 1
                    cost_arr[i] = cost_function_system(
                        problem.ineq_array, valves, conds,
                        problem.boundary_pressure_template, problem.incidence_matrix)

        print(f"  Adam sim {j}: completion={completion_index}, kicks={num_kick}, "
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
        plt.title("Adam - cost over time")
        plt.xlabel("Step")
        plt.ylabel("Cost")
        plt.plot(cost_arr)
        plt.show(block=True)

    return all_results


if __name__ == "__main__":
    run(seed=-1, num_steps=556, num_sim=1, show_plot=True)
