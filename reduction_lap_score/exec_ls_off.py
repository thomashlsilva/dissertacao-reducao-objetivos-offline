import numpy as np
import pandas as pd
from lap_score import LapScore


def main():
    number_dtlz = 7
    n_obj_orig = int(input("Enter the number of original objectives: "))

    # Change dtlz number
    pop_f = pd.read_csv(f"../obj_func_database/dtlz{number_dtlz}_{n_obj_orig}obj.csv",
                        header=None).to_numpy()

    ls = LapScore(pop_f)
    scores, selected_features = ls.lap_score()

    pd.DataFrame(scores).to_csv(
        f"./scores/dtlz{number_dtlz}/{n_obj_orig}obj_scores.csv",
        index=False, header=False)

    pd.DataFrame(selected_features).to_csv(
        f"./scores/dtlz{number_dtlz}/{n_obj_orig}obj_selected_features.csv",
        index=False, header=False)

    print(selected_features)
    print(np.flip(selected_features))


if __name__ == '__main__':
    main()
