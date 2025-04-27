import numpy as np
import pandas as pd
from pymoo.problems import get_problem


def main():
    k = 5
    n_obj_orig = int(input("Enter the number of original objectives: "))
    n_var = k + n_obj_orig - 1

    problem = get_problem("dtlz7", n_var=n_var, n_obj=n_obj_orig)

    # create initial data and set to the population object
    x = np.random.random((2500, problem.n_var))

    pop_f = problem.evaluate(x, return_values_of=["F"])
    pd.DataFrame(pop_f).to_csv(f"./obj_func_database/dtlz7_{n_obj_orig}obj.csv",
                               index=False, header=False)


if __name__ == '__main__':
    main()
