import numpy as np
import pandas as pd
import hvwfg
import datetime
from pymoo.indicators.hv import HV


def main():
    # get the start datetime
    st = datetime.datetime.now()
    n_obj_red_list = None
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

    pf_combined = pd.read_csv(f"../reduction_lap_score/dtlz1_elim/pareto_front/combined/{n_obj}obj_pf.csv", header=None).to_numpy()
    max_pf_combined = np.max(pf_combined, axis=0)
    # print(max_pf_combined)
    min_pf_combined = np.min(pf_combined, axis=0)

    # Load and combine Pareto fronts
    for n_obj_red in n_obj_red_list:
        pf_list = []
        vec_hv = []
        for i in range(21):
            pareto_restrito = []
            pf_list.append(pd.read_csv(f"../reduction_lap_score/dtlz1_elim/pareto_front/{n_obj}obj/{n_obj_red}red_pf{i}.csv",
                                       header=None).to_numpy())
            for j in range(300):
                if np.all(pf_list[i][j, :] <= max_pf_combined):
                    pareto_restrito.append(pf_list[i][j, :])

            pareto_restrito = np.array(pareto_restrito)
            mi_Pareto = np.tile(min_pf_combined, (pareto_restrito.shape[0], 1))
            ma_Pareto = np.tile(max_pf_combined, (pareto_restrito.shape[0], 1))
            pareto_restrito_normalizado = (pareto_restrito - mi_Pareto) / (ma_Pareto - mi_Pareto)

            HV_i = hvwfg.wfg(pareto_restrito_normalizado, np.ones(n_obj))
            vec_hv.append(HV_i)

        vec_hv = np.array(vec_hv)
        pd.DataFrame(vec_hv).to_csv(f"../reduction_lap_score/dtlz1_elim/hv/{n_obj}obj_{n_obj_red}red_hvwfg.csv",
                                    index=False, header=False)
    # get the end datetime
    et = datetime.datetime.now()
    # get execution time
    elapsed_time = et - st
    print('Start datetime:', st, 'seconds')
    print('End datetime:', et, 'seconds')
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == '__main__':
    main()
