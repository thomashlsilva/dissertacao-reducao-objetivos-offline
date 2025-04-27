from scipy import stats
import numpy as np
import pandas as pd
import sys


def main():
    number_dtlz = 7

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

    # Save the current standard output
    original_stdout = sys.stdout
    # Redirect standard output to a file
    sys.stdout = open(f'./dtlz{number_dtlz}_pca/wilcoxon_results/{n_obj}obj.txt', 'w')

    # IGD+ vector from original problem
    igd_orig = pd.read_csv(f"./dtlz{number_dtlz}_pca/igd_plus/{n_obj}obj_{n_obj}red_igd_plus.csv",
                           header=None).to_numpy()
    # Epsilon-indicator vector from original problem
    eind_orig = pd.read_csv(f"./dtlz{number_dtlz}_pca/eps_ind/{n_obj}obj_{n_obj}red_eind.csv",
                            header=None).to_numpy()

    for n_obj_red in n_obj_red_list:
        print('\nNumber of objectives:', n_obj_red)
        # Read IGD+ for all cases
        igd_arr = pd.read_csv(f"./dtlz{number_dtlz}_pca/igd_plus/{n_obj}obj_{n_obj_red}red_igd_plus.csv",
                              header=None).to_numpy()
        # Read epsilon-indicator for all cases
        eind_arr = pd.read_csv(f"./dtlz{number_dtlz}_pca/eps_ind/{n_obj}obj_{n_obj_red}red_eind.csv",
                               header=None).to_numpy()

        print('\n')
        # IGD+
        print('IGD+ mean: ', np.mean(igd_arr))
        print('IGD+ Wilcoxon rank-sum test: ',
              stats.ranksums(igd_arr, igd_orig, alternative='two-sided', axis=0))

        if np.array_equal(igd_arr, igd_orig):
            print("igd_arr and igd_orig are identical.")
        else:
            print("igd_arr and igd_orig are not identical.")

        print('\n')
        # Epsilon-indicator
        print('Epsilon-indicator mean: ', np.mean(eind_arr))
        print('Epsilon-indicator Wilcoxon rank-sum test: ',
              stats.ranksums(eind_arr, eind_orig, alternative='two-sided', axis=0))

        if np.array_equal(eind_arr, eind_orig):
            print("eind_arr and eind_orig are identical.")
        else:
            print("eind_arr and eind_orig are not identical.")

    # Restore the standard output
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()
