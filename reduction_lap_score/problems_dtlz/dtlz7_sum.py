import numpy as np
from pymoo.core.problem import Problem


class DTLZ7Sum(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, selected_features, k=5, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.selected_features = selected_features
        self.k = k

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)
    """
    def obj_func(self, X_, g, alpha=1):
        f = []

        # for i in range(0, self.n_obj):
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
    """

    def obj_func(self, x):
        # montar as funções normalmente, depois reordená-las
        # de acordo com o selected_features
        f = []

        for i in range(0, self.n_obj_orig - 1):
            f.append(x[:, i])
        f = np.column_stack(f)

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj_orig - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)

        f_orig = np.column_stack([f, (1 + g) * h])

        f_red = []
        for i in range(len(self.selected_features)):
            aux = self.selected_features[i]
            f_red.append(f_orig[:, aux])
        f_full = np.column_stack(f_red)

        num_cols_to_sum = self.n_obj_orig - self.n_obj_red + 1

        # select the functions to sum
        f_sum = np.sum(f_full[:, :num_cols_to_sum], axis=1, keepdims=True)

        # mix the added functions to the remaining original functions
        res = np.append(f_sum, f_full[:, num_cols_to_sum:], axis=1)
        return res

    def _evaluate(self, x, out, *args, **kwargs):
        # observe run_main_ls
        # out["F"] = np.flip(self.obj_func(x), axis=1)
        out["F"] = self.obj_func(x)
