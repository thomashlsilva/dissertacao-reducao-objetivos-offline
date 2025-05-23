import numpy as np
import pandas as pd
from pymoo.core.problem import Problem


class DTLZ1PCA(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, eigenvectors, k=5, execution_id=0, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.eigenvectors = eigenvectors
        self.k = k
        self.execution_id = execution_id  # Identificador para salvar arquivos

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)

    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj_orig):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)
        f_full = np.column_stack(f)

        # Normalizar a matriz usando Z-score pelo numpy
        mean = np.mean(f_full, axis=0)
        sd = np.std(f_full, axis=0, ddof=1)
        '''
        # Salvar média e desvio padrão com identificador da execução
        mean_file = f"dtlz1_pca/mean/{self.n_obj_orig}obj/{self.n_obj_red}red_exec{self.execution_id}.csv"
        sd_file = f"dtlz1_pca/sd/{self.n_obj_orig}obj/{self.n_obj_red}red_exec{self.execution_id}.csv"

        pd.DataFrame(mean).to_csv(mean_file, index=False, header=False)
        pd.DataFrame(sd).to_csv(sd_file, index=False, header=False)
        '''
        f_normalizada = (f_full - mean) / sd
        # Aplicar PCA
        f_pca = f_normalizada @ self.eigenvectors

        # Cortar a matriz original pelo número reduzido de objetivos
        return f_pca[:, :self.n_obj_red]

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj_orig - 1], x[:, self.n_obj_orig - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
