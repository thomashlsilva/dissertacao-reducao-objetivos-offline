from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import numpy as np
import pandas as pd
from problems_dtlz.dtlz1_pca import DTLZ1PCA
from problems_dtlz.dtlz2_pca import DTLZ2PCA
from problems_dtlz.dtlz3_pca import DTLZ3PCA
from problems_dtlz.dtlz4_pca import DTLZ4PCA
from problems_dtlz.dtlz5_pca import DTLZ5PCA
from problems_dtlz.dtlz6_pca import DTLZ6PCA
from problems_dtlz.dtlz7_pca import DTLZ7PCA


def main(arguments):
    # ALTERAR RUN_MAIN_LS
    number_dtlz = 1

    k = 5  # fixed
    n_obj_orig, n_obj_red, eigenvectors = arguments
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
        print('Number of objectives reduced: ', n_obj_red, ' Run number:', i)

        # Reduced problem
        problem = DTLZ1PCA(n_var=n_var, n_obj_orig=n_obj_orig, n_obj_red=n_obj_red, eigenvectors=eigenvectors)

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
            f"./dtlz{number_dtlz}_pca/decision_spaces/{n_obj_orig}obj/pop_red{n_obj_red}_x{i}.csv",
            index=False, header=False)

        # Expansion
        problem_orig = get_problem(f"dtlz{number_dtlz}", n_var=n_var, n_obj=n_obj_orig)
        pop_exp_f = problem_orig.evaluate(pop_x, return_values_of=["F"])

        # Calcular a média e o sd ao longo das linhas (axis=0)
        media = np.mean(pop_exp_f, axis=0)
        sd = np.std(pop_exp_f, axis=0, ddof=1)
        # Normalizar a matriz usando Z-score using numpy
        f_normalizada = (pop_exp_f - media) / sd
        # eigenvectors disposed in columns
        pop_exp_f_pca = f_normalizada @ eigenvectors

        # Saves the Pareto front for all runs
        pd.DataFrame(pop_exp_f_pca).to_csv(
            f"./dtlz{number_dtlz}_pca/pareto_front/{n_obj_orig}obj/{n_obj_red}red_pf{i}.csv",
            index=False, header=False)


def parse_arguments():
    pass


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
