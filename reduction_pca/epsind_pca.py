import numpy as np
import pandas as pd
import datetime
import winsound


def eps_ind_add(A, B):
    aux = []
    for i in range(len(B)):
        aux.append(np.min(np.max(A - np.tile(B[i], (A.shape[0], 1)), axis=1)))
    return max(aux)


def main():
    number_dtlz = 7

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

    pf_combined = pd.read_csv(f"./dtlz{number_dtlz}_pca/pareto_front/combined/{n_obj}obj_pf.csv", header=None).to_numpy()

    for n_obj_red in n_obj_red_list:
        print('\nNumber of objectives:', n_obj_red)
        pf_list = []
        eind = []
        for j in range(21):
            pf_list.append(pd.read_csv(f"./dtlz{number_dtlz}_pca/pareto_front/{n_obj}obj/{n_obj_red}red_pf{j}.csv",
                                       header=None).to_numpy())
            eind.append(eps_ind_add(pf_list[j], pf_combined))

        eind_arr = np.array(eind)
        pd.DataFrame(eind_arr).to_csv(f"dtlz{number_dtlz}_pca/eps_ind/{n_obj}obj_{n_obj_red}red_eind.csv",
                                      index=False, header=False)

    # get the end datetime
    et = datetime.datetime.now()
    # get execution time
    elapsed_time = et - st
    print('Start datetime:', st, 'seconds')
    print('End datetime:', et, 'seconds')
    print('Execution time:', elapsed_time, 'seconds')

    duration = 750  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


if __name__ == '__main__':
    main()
