from pymoo.indicators.igd_plus import IGDPlus
import numpy as np
import pandas as pd
import datetime
import multiprocessing
import winsound


def compute_igd(n_obj, n_obj_red):
    pf_combined = pd.read_csv(f"./dtlz1_pca/pareto_front/combined/{n_obj}obj_pf.csv", header=None).to_numpy()
    ind = IGDPlus(pf_combined)

    pf_list = []
    igd = []
    for j in range(21):
        print('Number of objectives:', n_obj_red, 'PF:', j)
        pf_list.append(
            pd.read_csv(f"./dtlz1_pca/pareto_front/{n_obj}obj/{n_obj_red}red_pf{j}.csv", header=None).to_numpy())
        igd.append(ind(pf_list[j]))

    igd_arr = np.array(igd)
    pd.DataFrame(igd_arr).to_csv(f"dtlz1_pca/igd_plus/{n_obj}obj_{n_obj_red}red_igd_plus.csv",
                                 index=False, header=False)
    print('\nIGD+ finished:', n_obj_red)


def main():
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

    # Create a multiprocessing pool with the number of processes equal to the number of n_obj_red values
    pool = multiprocessing.Pool(2)

    for n_obj_red in n_obj_red_list:
        pool.apply_async(compute_igd, (n_obj, n_obj_red))

    pool.close()
    pool.join()

    et = datetime.datetime.now()
    elapsed_time = et - st
    print('Start datetime:', st, 'seconds')
    print('End datetime:', et, 'seconds')
    print('Execution time:', elapsed_time, 'seconds')
    duration = 750  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


if __name__ == '__main__':
    main()
