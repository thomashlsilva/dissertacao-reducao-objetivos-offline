import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from problems_dtlz.dtlz1_elim import DTLZ1Elim
from problems_dtlz.dtlz1_sum import DTLZ1Sum
from problems_dtlz.dtlz2_sum import DTLZ2Sum
from problems_dtlz.dtlz3_sum import DTLZ3Sum
from problems_dtlz.dtlz4_sum import DTLZ4Sum
from problems_dtlz.dtlz5_sum import DTLZ5Sum
from problems_dtlz.dtlz6_sum import DTLZ6Sum
from problems_dtlz.dtlz7_sum import DTLZ7Sum


def main(arguments):
    # ALTERAR RUN_MAIN_LS
    number_dtlz = 7

    k = 5  # fixed
    n_obj_orig, n_obj_red, selected_features = arguments
    n_var = k + n_obj_orig - 1

    np.seterr(invalid='ignore')

    # Number of evaluations
    termination = get_termination("n_eval", 30000)

    # Number of individuals
    pop_size = 300

    # Initial population
    np.random.seed(1)
    x = np.random.random((pop_size, n_var))

    # 21 algorithm runs
    for i in range(21):
        print('Number of objectives: ', n_obj_red, ' Run number:', i)

        problem = DTLZ7Sum(n_var=n_var, n_obj_orig=n_obj_orig, n_obj_red=n_obj_red,
                           selected_features=selected_features)

        ref_dirs = get_reference_directions("energy", n_obj_red, pop_size, seed=i)

        algorithm = NSGA3(pop_size=pop_size,
                          ref_dirs=ref_dirs,
                          sampling=x)

        res_problem = minimize(problem,
                               algorithm,
                               seed=i,
                               termination=termination,
                               verbose=False)

        pop_x = res_problem.pop.get("X")

        # Saves decision spaces for all runs
        pd.DataFrame(pop_x).to_csv(
            f"./dtlz{number_dtlz}_sum/decision_spaces/{n_obj_orig}obj/pop_red{n_obj_red}_x{i}.csv",
            index=False, header=False)

        # Expansion
        problem_orig = get_problem(f"dtlz{number_dtlz}", n_var=n_var, n_obj=n_obj_orig)
        pop_exp_f = problem_orig.evaluate(pop_x, return_values_of=["F"])

        # Saves Pareto front for all runs
        pd.DataFrame(pop_exp_f).to_csv(
            f"./dtlz{number_dtlz}_sum/pareto_front/{n_obj_orig}obj/{n_obj_red}red_pf{i}.csv",
            index=False, header=False)




def parse_arguments():
    pass


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
