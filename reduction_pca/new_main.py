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
    # Alterar também no RUN_MAIN_PCA
    number_dtlz = 7

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

        # Reduced problem with execution_id
        problem = DTLZ7PCA(n_var=n_var, n_obj_orig=n_obj_orig, n_obj_red=n_obj_red, eigenvectors=eigenvectors,
                           execution_id=i)

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
        pop_pca = res_problem.pop.get("F")

        # Save decision spaces for all runs
        pd.DataFrame(pop_x).to_csv(
            f"./dtlz{number_dtlz}_pca/decision_spaces/{n_obj_orig}obj/pop_red{n_obj_red}_x{i}.csv",
            index=False, header=False)
        '''
        # Reverter a normalização e converter de volta para o espaço original
        # Leitura dos arquivos de média e desvio padrão
        sd = pd.read_csv(f"dtlz1_pca/sd/{n_obj_orig}obj/{n_obj_red}red_exec{i}.csv",
                        header=None).to_numpy()
        mean = pd.read_csv(f"dtlz1_pca/mean/{n_obj_orig}obj/{n_obj_red}red_exec{i}.csv",
                          header=None).to_numpy()
        mean_col = mean.flatten()  # ou mean.squeeze()
        sd_col = sd.flatten()  # ou sd.squeeze()
        
        # Aplicar PCA inversa e reverter a normalização Z-score
        pop_f = pop_pca * sd_col + mean_col
        '''
        # Aplicar PCA inversa
        pop_f = pop_pca @ np.linalg.pinv(eigenvectors[:, :pop_pca.shape[1]])

        # Save the Pareto front for all runs
        pd.DataFrame(pop_f).to_csv(f"./dtlz{number_dtlz}_pca/pareto_front/{n_obj_orig}obj/{n_obj_red}red_pf{i}.csv",
                                   index=False, header=False)


def parse_arguments():
    pass


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
