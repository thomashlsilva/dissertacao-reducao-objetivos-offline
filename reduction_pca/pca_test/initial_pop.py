import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from reduction_pca.pca_sueli import PCAReduction


def initial_pop():
    # Initial parameters
    k = 5
    n_obj = 2
    number_dtlz = 1
    n_var = k + n_obj - 1

    # Problem declaration
    problem = get_problem(f"dtlz{number_dtlz}", n_var=n_var, n_obj=n_obj)

    # Create initial data and set to the population object
    np.random.seed(1)
    pop_x = np.random.random((2500, n_var))

    # Evaluate the decision space and saves the objective space
    pop_f = problem.evaluate(pop_x, return_values_of=["F"])

    # PCA with correlation matrix results in eigenvalues and eigenvectors
    pca_call = PCAReduction(pop_f)
    explained_variance_ratio, eigenvectors = pca_call.pca()

    pd.DataFrame(explained_variance_ratio).to_csv(
        f"./dtlz{number_dtlz}_{n_obj}obj_expvar_ratio.csv",
        index=False, header=False)

    pd.DataFrame(eigenvectors).to_csv(
        f"./dtlz{number_dtlz}_{n_obj}obj_eigenvectors.csv",
        index=False, header=False)


if __name__ == '__main__':
    initial_pop()
