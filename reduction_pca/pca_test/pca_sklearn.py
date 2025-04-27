import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class PCAReduction:
    def __init__(self, pop_f):
        # MinMaxScaler shows lower changes on the original data
        self.scaled_pop_f = MinMaxScaler().fit_transform(pop_f)

        # Z-Score using numpy
        # Calcular a média e o sd ao longo das linhas (axis=0)
        # media = np.mean(pop_f, axis=0)
        # sd = np.std(pop_f, axis=0, ddof=1)

        # Normalizar a matriz usando Z-score
        # self.scaled_pop_f = (pop_f - media) / sd

    def pca(self):
        pca = PCA()
        pca.fit(self.scaled_pop_f)
        exp_var_ratio = pca.explained_variance_ratio_
        components = pca.components_.T  # Transpose the components

        return exp_var_ratio, components
