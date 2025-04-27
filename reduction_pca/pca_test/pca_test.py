from reduction_pca.pca_sueli import PCAReduction
import numpy as np
import pandas as pd


def main():
    matriz = np.array([[9893, 564, 17689], [8776, 389, 17359], [13572, 1103, 18597], [6455, 743, 8745],
                        [5129, 203, 14397], [5432, 215, 3467], [3807, 385, 4679], [3423, 187, 6754],
                        [3708, 127, 2275], [3294, 297, 6754], [5433, 432, 5589], [6287, 451, 8972]])
    print(matriz)
    pca_call = PCAReduction(matriz)
    explained_variance_ratio, eigenvectors = pca_call.pca()

    pd.DataFrame(explained_variance_ratio).to_csv(f"../../pca_test_expvar_ratio.csv", index=False, header=False)

    pd.DataFrame(eigenvectors).to_csv(f"../../pca_test_eigenvectors.csv", index=False, header=False)

    # Calcular a média e o desvio padrão ao longo das linhas (axis=0)
    media = np.mean(matriz, axis=0)
    desvio_padrao = np.std(matriz, axis=0, ddof=1)

    # Normalizar a matriz usando Z-score
    matriz_normalizada = (matriz - media) / desvio_padrao

    res = matriz_normalizada @ eigenvectors
    print(res)
    res2 = matriz @ eigenvectors
    print(res2)

    eigenvectors_Sueli = np.array([[0.61670267, 0.00126721, 0.78719515],
                                   [0.55679445, 0.70619694, -0.43733949],
                                   [0.556469, -0.70801432, -0.43480796]])

    res_Sueli = matriz_normalizada @ eigenvectors_Sueli

    print(np.allclose(res, res_Sueli))


if __name__ == '__main__':
    main()






