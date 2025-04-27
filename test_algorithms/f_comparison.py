import pandas as pd
import numpy as np


def main():
    x = pd.read_csv(f"reduction_lap_score/dtlz2_sum/decision_spaces/5obj/pop_red5_x12.csv", header=None).to_numpy()
    n_obj = 5
    n_obj_red = 5
    alpha = 1
    selected_features = np.array([0, 1, 2, 3, 4])

    X_, X_M = x[:, :n_obj - 1], x[:, n_obj - 1:]

    g2 = np.sum(np.square(X_M - 0.5), axis=1)
    """
    f = []

    for i in range(len(selected_features)):
        aux = selected_features[i]
        _f = (1 + g2)
        _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - aux], alpha) * np.pi / 2.0), axis=1)
        if aux > 0:
            _f *= np.sin(np.power(X_[:, X_.shape[1] - aux], alpha) * np.pi / 2.0)
        f.append(_f)

    f_full = np.column_stack(f)

    num_cols_to_sum = n_obj - n_obj_red + 1

    # select the functions to sum
    f_sum = np.sum(f_full[:, :num_cols_to_sum], axis=1, keepdims=True)

    # mix the added functions to the remaining original functions
    res = np.append(f_sum, f_full[:, num_cols_to_sum:], axis=1)

    pd.DataFrame(res).to_csv(f"./result_mydtlz2sum.csv", index=False, header=False)
    """
    f = []

    for j in range(0, n_obj):
        _f = (1 + g2)
        _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - j], alpha) * np.pi / 2.0), axis=1)
        if j > 0:
            _f *= np.sin(np.power(X_[:, X_.shape[1] - j], alpha) * np.pi / 2.0)

        f.append(_f)

    f = np.column_stack(f)

    pd.DataFrame(f).to_csv(f"./result_dtlz2.csv", index=False, header=False)



if __name__ == '__main__':
    main()
