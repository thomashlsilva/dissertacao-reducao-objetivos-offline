import pandas as pd
import numpy as np
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem


def main():
    pf_list = []
    pf_list_orig = []
    vec = []
    vec_orig = []
    n_obj = 5
    n_obj_red = 5
    pop_size = 300
    k = 5  # fixed
    n_var = k + n_obj - 1

    for j in range(21):
        print('\nNumber of objectives:', n_obj_red, '\nPF:', j)
        pf_list.append(pd.read_csv(f"../reduction_lap_score/dtlz3_sum/pareto_front/{n_obj}obj/{n_obj_red}red_pf{j}.csv",
                                   header=None).to_numpy())
        pf_list_orig.append(pd.read_csv(f"../original_problem_pf/dtlz3_{n_obj}obj_pf{j}.csv",
                                        header=None).to_numpy())

        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=j)
        # The pareto front of a scaled problem
        pf = get_problem("dtlz3", n_obj=n_obj, n_var=n_var).pareto_front(ref_dirs)
        ind = IGDPlus(pf)
        vec.append(ind(pf_list[j]))
        vec_orig.append(ind(pf_list_orig[j]))
        """
        igd_plus = IGDPlus(pf)
        for index, elem in enumerate(pf_list):
            ind[index] = igd_plus(elem)
        """

    print("IGD+", np.array(vec))
    print('IGD+ mean', np.mean(np.array(vec)))

    print("IGD+ Original", np.array(vec_orig))
    print('IGD+ Original mean', np.mean(np.array(vec_orig)))

    print(np.allclose(vec, vec_orig))

if __name__ == '__main__':
    main()
