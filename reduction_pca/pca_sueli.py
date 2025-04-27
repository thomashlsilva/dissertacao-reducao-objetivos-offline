import numpy as np
from numpy.linalg import eig
import pandas as pd


class PCAReduction:
    def __init__(self, df):
        self.df = pd.DataFrame(df)

    def pca(self):
        # Calcula a matriz de correlação do DataFrame
        corr_matrix = self.df.corr()

        # Cálculo dos Autovalores e Autovetores
        e_values, e_vectors = eig(corr_matrix)

        # Ordena os autovalores e autovetores em ordem decrescente
        idx = e_values.argsort()[::-1]

        e_values = e_values[idx]
        e_vectors = e_vectors[:, idx]

        # Calcula a razão de variância explicada para cada componente principal
        e_values_sum = np.sum(e_values)
        explained_variance_ratio = e_values / e_values_sum

        return explained_variance_ratio, e_vectors
