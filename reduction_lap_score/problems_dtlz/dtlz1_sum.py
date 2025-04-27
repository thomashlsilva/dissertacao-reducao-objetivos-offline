import numpy as np
from pymoo.core.problem import Problem


class DTLZ1Sum(Problem):
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

        for i in range(len(self.selected_features)):
            aux = self.selected_features[i]
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - aux], axis=1)
            if aux > 0:
                _f *= 1 - X_[:, X_.shape[1] - aux]
            f.append(_f)
        f_full = np.column_stack(f)

        num_cols_to_sum = self.n_obj_orig - self.n_obj_red + 1

        # select the functions to sum
        f_sum = np.sum(f_full[:, :num_cols_to_sum], axis=1, keepdims=True)

        # mix the added functions to the remaining original functions
        res = np.append(f_sum, f_full[:, num_cols_to_sum:], axis=1)

        """
        pd.DataFrame(f_full).to_csv("./obj_func_database/f_full_orig.csv", index=False, header=False)
        f_full = MinMaxScaler().fit_transform(f_full)
        f_sum = np.cumsum(f_full[:, self.n_obj_red - 1:], axis=1)
        f_sum = np.cumsum(f_full[:, self.n_obj_red - 1:self.n_obj_orig - 2], axis=1)
        alterei p/ n_obj_orig-2 no caso com funcoes cortadas
        res = np.append(f_full[:, :self.n_obj_red - 1], f_sum[:, [-1]], axis=1)

        pd.DataFrame(f_full).to_csv("./obj_func_database/f_full.csv", index=False, header=False)
        pd.DataFrame(f_sum).to_csv("./obj_func_database/f_acumulative.csv", index=False,header=False)
        pd.DataFrame(res).to_csv("./obj_func_database/f_final.csv", index=False, header=False)
        """
        return res

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj_orig - 1], x[:, self.n_obj_orig - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
