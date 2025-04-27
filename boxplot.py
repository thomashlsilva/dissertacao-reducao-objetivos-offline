import pandas as pd
import matplotlib.pyplot as plt


def main():
    number_dtlz = 7
    n_obj = int(input("Enter the number of original objectives: "))

    if n_obj == 5:
        n_obj_red_list = [5, 4, 3, 2]
    elif n_obj == 10:
        n_obj_red_list = [10, 8, 6, 4, 2]
    elif n_obj == 15:
        n_obj_red_list = [15, 12, 9, 6, 3, 2]
    elif n_obj == 20:
        n_obj_red_list = [20, 16, 12, 8, 4, 2]
    elif n_obj == 30:
        n_obj_red_list = [30, 25, 20, 15, 10, 5, 2]
    else:
        print("Invalid number of objectives!")
        return

    igd_data_dict = {n_obj_red: [] for n_obj_red in n_obj_red_list}
    eps_data_dict = {n_obj_red: [] for n_obj_red in n_obj_red_list}

    for n_obj_red in n_obj_red_list:
        for j in range(21):
            igd_data = pd.read_csv(
                f"reduction_pca/dtlz{number_dtlz}_pca/igd_plus/{n_obj}obj_{n_obj_red}red_igd_plus.csv",
                header=None).squeeze().tolist()
            eps_data = pd.read_csv(
                f"reduction_pca/dtlz{number_dtlz}_pca/eps_ind/{n_obj}obj_{n_obj_red}red_eind.csv",
                header=None).squeeze().tolist()

            igd_data_dict[n_obj_red].append(igd_data)
            eps_data_dict[n_obj_red].append(eps_data)

    # Transforma listas de listas em listas planas
    for key in igd_data_dict:
        igd_data_dict[key] = [item for sublist in igd_data_dict[key] for item in sublist]
    for key in eps_data_dict:
        eps_data_dict[key] = [item for sublist in eps_data_dict[key] for item in sublist]

    igd_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in igd_data_dict.items()]))
    eps_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in eps_data_dict.items()]))

    # Cria figura para IGD
    fig_igd, ax_igd = plt.subplots(figsize=(5, 4))
    igd_df.plot(kind='box', ax=ax_igd, color='black', medianprops=dict(color='orange'))
    ax_igd.set_xlabel('Número de Objetivos')
    ax_igd.set_ylabel('Valores de IGD+')
    plt.tight_layout()
    plt.show()
    fig_igd.savefig(f"reduction_pca/dtlz{number_dtlz}_pca/plots/{n_obj}orig_igd.pdf")

    # Cria figura para Epsilon
    fig_eps, ax_eps = plt.subplots(figsize=(5, 4))
    eps_df.plot(kind='box', ax=ax_eps, color='black', medianprops=dict(color='orange'))
    ax_eps.set_xlabel('Número de Objetivos')
    ax_eps.set_ylabel('Valores de Epsilon Aditivo')
    plt.tight_layout()
    plt.show()
    fig_eps.savefig(f"reduction_pca/dtlz{number_dtlz}_pca/plots/{n_obj}orig_eps.pdf")


if __name__ == '__main__':
    main()
