import numpy as np
import pandas as pd
import hvwfg
import multiprocessing
import datetime
from functools import partial


def compute_hv(n_obj, n_obj_red, max_pf_combined, min_pf_combined, index):
    pf = pd.read_csv(f"./dtlz1_elim/pareto_front/{n_obj}obj/{n_obj_red}red_pf{index}.csv",
                     header=None).to_numpy()

    pareto_restrito = [row for row in pf if np.all(row <= max_pf_combined)]
    pareto_restrito = np.array(pareto_restrito)

    mi_Pareto = np.tile(min_pf_combined, (pareto_restrito.shape[0], 1))
    ma_Pareto = np.tile(max_pf_combined, (pareto_restrito.shape[0], 1))

    pareto_restrito_normalizado = (pareto_restrito - mi_Pareto) / (ma_Pareto - mi_Pareto)

    HV_i = hvwfg.wfg(pareto_restrito_normalizado, np.ones(n_obj))
    return HV_i


def main():
    # get the start datetime
    st = datetime.datetime.now()
    n_obj = int(input("Enter the number of original objectives: "))
    if n_obj == 5:
        n_obj_red_list = [5, 4, 3, 2]
    elif n_obj == 10:
        n_obj_red_list = [10, 8, 6, 4, 2]
    elif n_obj == 15:
        n_obj_red_list = [15, 12, 9, 6, 3, 2]
    elif n_obj == 20:
        n_obj_red_list = [20, 16, 12, 8, 4, 2]
    elif n_obj == 30:
        n_obj_red_list = [30, 25, 20, 15, 10, 5, 2]
    else:
        print("Invalid number of objectives!")
        return

    pf_combined = pd.read_csv(f"./dtlz1_elim/pareto_front/combined/{n_obj}obj_pf.csv", header=None).to_numpy()
    max_pf_combined = np.max(pf_combined, axis=0)
    min_pf_combined = np.min(pf_combined, axis=0)

    pool = multiprocessing.Pool(4)

    for n_obj_red in n_obj_red_list:
        partial_compute_hv = partial(compute_hv, n_obj, n_obj_red, max_pf_combined, min_pf_combined)
        vec_hv = pool.map(partial_compute_hv, range(21))

        vec_hv = np.array(vec_hv)
        pd.DataFrame(vec_hv).to_csv(f"./dtlz1_elim/hv/{n_obj}obj_{n_obj_red}red_hvwfg.csv",
                                    index=False, header=False)

    pool.close()
    pool.join()

    # get the end datetime
    et = datetime.datetime.now()
    # get execution time
    elapsed_time = et - st
    print('Start datetime:', st, 'seconds')
    print('End datetime:', et, 'seconds')
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == '__main__':
    main()
