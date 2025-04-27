import numpy as np
from pymoo.core.problem import Problem


class DTLZ7PCA(Problem):
    def __init__(self, n_var, n_obj_orig, n_obj_red, eigenvectors, k=5, **kwargs):
        self.n_obj_orig = n_obj_orig
        self.n_obj_red = n_obj_red
        self.eigenvectors = eigenvectors
        self.k = k

        super().__init__(n_var=n_var, n_obj=n_obj_red, xl=0, xu=1, vtype=float, **kwargs)

    def obj_func(self, x):
        f = []

        for i in range(0, self.n_obj_orig - 1):
            f.append(x[:, i])
        f = np.column_stack(f)

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj_orig - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)

        f_full = np.column_stack([f, (1 + g) * h])

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
        out["F"] = self.obj_func(x)
