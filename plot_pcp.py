import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pymoo.visualization.pcp import PCP


def main():
    number_dtlz = 1

    n_obj = 5  # 5 or 10
    pf_orig = 1

    n_obj_red = 2  # which reduction is better?
    pf_red = 1

    # pf value depends on the approximation of the mean
    F1 = pd.read_csv(f"reduction_lap_score/dtlz{number_dtlz}_sum/pareto_front/{n_obj}obj/{n_obj}red_pf{pf_orig}.csv",
                     header=None).to_numpy()
    scaled_F1 = MinMaxScaler().fit_transform(F1)

    F2 = pd.read_csv(f"reduction_lap_score/dtlz{number_dtlz}_sum/pareto_front/{n_obj}obj/{n_obj_red}red_pf{pf_red}.csv",
                     header=None).to_numpy()
    scaled_F2 = MinMaxScaler().fit_transform(F2)

    pcp1 = PCP(tight_layout=True, figsize=(8, 6)).add(scaled_F1).show()
    pcp2 = PCP(tight_layout=True, figsize=(8, 6)).add(scaled_F2).show()

    pcp1.save(f"reduction_lap_score/dtlz{number_dtlz}_sum/plots/{n_obj}orig_pf{pf_orig}.pdf")
    pcp2.save(f"reduction_lap_score/dtlz{number_dtlz}_sum/plots/{n_obj}orig_{n_obj_red}red_pf{pf_red}.pdf")


if __name__ == '__main__':
    main()
