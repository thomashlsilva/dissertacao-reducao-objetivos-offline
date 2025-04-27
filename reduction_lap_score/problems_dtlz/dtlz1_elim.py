import numpy as np
from pymoo.core.problem import Problem


class DTLZ1Elim(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, selected_features, k=5, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.selected_features = selected_features
        self.k = k

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)

    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def obj_func(self, X_, g):
        f = []
        num_cols_to_elim = self.n_obj_orig - self.n_obj_red
        vec = self.selected_features[num_cols_to_elim:]
        for i in range(len(vec)):
            aux = vec[i]
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - aux], axis=1)
            if aux > 0:
                _f *= 1 - X_[:, X_.shape[1] - aux]
            f.append(_f)

        return np.column_stack(f)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj_orig - 1], x[:, self.n_obj_orig - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
        # out["F"] = np.flip(self.obj_func(X_, g), axis=1)
