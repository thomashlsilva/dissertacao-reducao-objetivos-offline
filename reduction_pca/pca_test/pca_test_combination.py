import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.igd_plus import IGDPlus

from reduction_pca.problems_dtlz.dtlz1_pca import DTLZ1PCA


def eps_ind_add(A, B):
    aux = []
    for i in range(len(B)):
        aux.append(np.min(np.max(A - np.tile(B[i], (A.shape[0], 1)), axis=1)))
    return max(aux)


def pca_test_combination():
    k = 5
    n_obj = 2
    number_dtlz = 1
    n_var = k + n_obj - 1

    # Number of evaluations
    termination = get_termination("n_eval", 30000)

    # Number of individuals
    pop_size = 300

    # Sampling
    np.random.seed(1)
    x = np.random.random((pop_size, n_var))

    eigenvectors_1 = pd.read_csv(f"./pca_test/dtlz{number_dtlz}_{n_obj}obj_eigenvectors.csv", header=None).to_numpy()

    eigenvectors_2 = eigenvectors_1 * np.array([[-1, 1]])
    eigenvectors_3 = eigenvectors_1 * np.array([[1, -1]])
    eigenvectors_4 = eigenvectors_1 * np.array([[-1, -1]])

    # Original problem
    # problem = get_problem("dtlz1", n_var=n_var, n_obj=n_obj)

    # Problem with PCA
    problem = DTLZ1PCA(n_var=n_var, n_obj_orig=n_obj, n_obj_red=2, eigenvectors=eigenvectors_4)

    ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=1)

    algorithm = NSGA3(pop_size=pop_size,
                      ref_dirs=ref_dirs,
                      sampling=x)

    res_problem = minimize(problem,
                           algorithm,
                           seed=1,
                           termination=termination,
                           verbose=True)

    pop_f = res_problem.pop.get("F")
    pd.DataFrame(pop_f).to_csv(f"./pca_test/dtlz{number_dtlz}_{n_obj}obj_pf4.csv",
                               index=False, header=False)

    pf = pd.read_csv(f"./pca_test/dtlz{number_dtlz}_{n_obj}obj_pf0.csv", header=None).to_numpy()

    ind = IGDPlus(pf)
    print(ind(pop_f))
    print(eps_ind_add(pop_f, pf))


if __name__ == '__main__':
    pca_test_combination()
