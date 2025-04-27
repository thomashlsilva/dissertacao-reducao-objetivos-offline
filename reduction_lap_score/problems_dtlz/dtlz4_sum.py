import numpy as np
from pymoo.core.problem import Problem


class DTLZ4Sum(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, selected_features, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.selected_features = selected_features

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=100):
        f = []

        for i in range(len(self.selected_features)):
            aux = self.selected_features[i]
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - aux], alpha) * np.pi / 2.0), axis=1)
            if aux > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - aux], alpha) * np.pi / 2.0)
            f.append(_f)
        f_full = np.column_stack(f)

        num_cols_to_sum = self.n_obj_orig - self.n_obj_red + 1

        # select the functions to sum
        f_sum = np.sum(f_full[:, :num_cols_to_sum], axis=1, keepdims=True)

        # mix the added functions to the remaining original functions
        res = np.append(f_sum, f_full[:, num_cols_to_sum:], axis=1)

        return res

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj_orig - 1], x[:, self.n_obj_orig - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=100)
