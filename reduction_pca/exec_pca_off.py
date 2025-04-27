# from pca_sklearn import PCAReduction
from pca_sueli import PCAReduction
import pandas as pd


def main():
    number_dtlz = 1
    n_obj_orig = int(input("Enter the number of original objectives: "))

    pop_f = pd.read_csv(
        f"../obj_func_database/dtlz{number_dtlz}_{n_obj_orig}obj.csv",
        header=None).to_numpy()

    pca_call = PCAReduction(pop_f)
    explained_variance_ratio, eigenvectors = pca_call.pca()

    pd.DataFrame(explained_variance_ratio).to_csv(
        f"./eigenvectors/dtlz{number_dtlz}/{n_obj_orig}obj_expvar_ratio.csv",
        index=False, header=False)

    pd.DataFrame(eigenvectors).to_csv(
        f"./eigenvectors/dtlz{number_dtlz}/{n_obj_orig}obj_eigenvectors.csv",
        index=False, header=False)


if __name__ == '__main__':
    main()
