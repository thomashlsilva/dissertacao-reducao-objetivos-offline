import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class PCAReduction:
    def __init__(self, df, num_fea):
        self.df = df
        self.scaled_data = MinMaxScaler().fit_transform(df)
        self.num_fea = num_fea

    def pca(self):
        pca_dtlz1 = PCA(n_components=self.num_fea)
        #pca_dtlz1.fit_transform(self.scaled_data)
        pca_dtlz1.fit_transform(self.df)

        # centering the data does not affect the pca result: principalComponents_dtlz1_not_centered =
        # pca_dtlz1.fit_transform(self.scaled_data + self.scaled_data.mean(axis=0))

        # sd_cen = self.scaled_data - self.scaled_data.mean(axis=0)
        # print(np.matmul(sd_cen, np.transpose(pca_dtlz1.components_)))
        # print(np.allclose(principalComponents_dtlz1, np.matmul(sd_cen, np.transpose(pca_dtlz1.components_))))

        eigenvectors = np.transpose(pca_dtlz1.components_)
        # It will provide you with the amount of information or variance each principal component holds after
        # projecting the data to a lower dimensional subspace.
        print(pca_dtlz1.explained_variance_ratio_)

        return eigenvectors
