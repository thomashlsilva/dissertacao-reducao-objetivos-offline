import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import sys
import winsound

sys.path.insert(0, "../reduction_lap_score/problems_dtlz")

from dtlz7_sum import DTLZ7Sum


def main():
    # vec = []
    n_obj = 5
    n_obj_red = 4
    k = 5  # fixed
    n_var = k + n_obj - 1
    selected_features = np.array([4, 3, 2, 1, 0])
    # ref_point = np.array([5.0, 5.0, 5.0])

    # Number of evaluations
    termination = get_termination("n_eval", 30000)

    # Number of individuals
    pop_size = 300

    # Initial population
    np.random.seed(1)
    x = np.random.random((pop_size, n_var))

    # Original problem
    problem = get_problem("dtlz7", n_var=n_var, n_obj=n_obj)

    # Reduced problem
    problem = DTLZ7Sum(n_var=n_var, n_obj_orig=n_obj, n_obj_red=n_obj_red, selected_features=selected_features)

    pop_gen = problem.evaluate(x, return_values_of=["F"])
    pd.DataFrame(pop_gen).to_csv(f"../DTLZ7_{n_obj}obj_PF_1gen_test.csv",
                                 index=False, header=False)
"""
    # 21 algorithm runs
    for i in range(21):
        print('Number of objectives: ', n_obj, ' Run number:', i)

        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=i)

        algorithm = NSGA3(pop_size=pop_size,
                          ref_dirs=ref_dirs,
                          sampling=x)

        res_problem = minimize(problem,
                               algorithm,
                               seed=i,
                               termination=termination,
                               verbose=True)

        # pop_x = res_problem.pop.get("X")
        pop_x = res_problem.pop.get("X")
        # pd.DataFrame(pop_x).to_csv(f"../DTLZ1_{n_obj}obj_DE.csv", index=False, header=False)
        # pd.DataFrame(
        #     pop_x).to_csv(f"../reduction_lap_score/dtlz7_sum/decision_spaces/{n_obj}obj/pop_red{n_obj_red}_x{i}.csv",
        #                  index=False, header=False)

        pop_f = res_problem.pop.get("F")
        # pd.DataFrame(pop_f).to_csv(f"../DTLZ1_{n_obj}obj_PF.csv", index=False, header=False)
        # pd.DataFrame(
        #     pop_f).to_csv(f"../reduction_lap_score/dtlz7_sum/pareto_front/{n_obj}obj/{n_obj_red}red_pf{i}.csv",
        #                       index=False, header=False)

        # The pareto front of a scaled problem
        # pf = get_problem("dtlz1", n_obj=n_obj, n_var=n_var).pareto_front(ref_dirs)

        # ind = IGDPlus(pf)
        # vec.append(ind(pop_f))

        # ind = HV(ref_point=ref_point)
        # vec.append(ind(pop_f))

    # print("IGD+", np.array(vec))
    # print('IGD+ mean', np.mean(np.array(vec)))

    # print("HV", np.array(vec))
    # print('HV mean', np.mean(np.array(vec)))

    duration = 750  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
"""

if __name__ == '__main__':
    main()
