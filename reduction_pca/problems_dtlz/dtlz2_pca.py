import numpy as np
from pymoo.core.problem import Problem


class DTLZ2PCA(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, eigenvectors, k=5, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.eigenvectors = eigenvectors
        self.k = k

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj_orig):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)
        f_full = np.column_stack(f)

        # Normalizar a matriz usando Z-score pelo numpy
        mean = np.mean(f_full, axis=0)
        sd = np.std(f_full, axis=0, ddof=1)

        f_normalizada = (f_full - mean) / sd
        # Aplicar PCA
        f_pca = f_normalizada @ self.eigenvectors

        # Cortar a matriz original pelo número reduzido de objetivos
        return f_pca[:, :self.n_obj_red]

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj_orig - 1], x[:, self.n_obj_orig - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)
