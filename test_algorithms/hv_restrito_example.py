import numpy as np
from hvwfg import wfg
import pareto


def main():
    # Suponha que uma execução de um MOEA (MOEA1) obteve uma frente de Pareto
    # com 10 soluções em um problema com 5 funções objetivo, todas as entradas
    # com valores entre -10 e 10 e algumas entradas 100000, muito destoantes
    # desse intervalo.
    MOEA1 = np.array([[4.4795, 5.0253, 2.3209, -2.9668, -8.4829],
                      [-6.7478, -4.8981, -4.9144, 6.6166, -8.9210],
                      [-8.8076, 3.9815, -5.1295, 0.9945, 100000],
                      [-7.6200, 7.8181, 6.8143, 100000, 5.5833],
                      [-0.0327, 100000, -3.0003, 5.0746, -7.4019],
                      [100000, 0.9443, -6.0681, 5.1440, 1.3765],
                      [-3.1923, -7.2275, -4.9783, -4.2832, -0.6122],
                      [1.7054, -4.8498, 100000, -2.3911, -9.7620],
                      [-5.5238, -7.0141, -0.5342, 1.3564, -3.2575],
                      [2.8864, -5.8452, -3.7780, 1.8979, -8.2897]])

    # Suponha que uma execução de outro MOEA (MOEA2) obteve uma frente de Pareto
    # com 10 soluções em um problema com 5 funções objetivo, todas as entradas
    # com valores entre -10 e 10.
    MOEA2 = np.array([[-9.7564, -0.9892, -7.8669, -1.3717, 7.0606],
                      [-8.7757, -5.4205, -9.9073, -6.3631, -2.9810],
                      [-6.6870, -6.9524, -6.3461, -7.0892, -8.9638],
                      [2.0396, -6.5163, -7.3739, -7.2786, -8.4807],
                      [-4.7406, 0.7668, -8.3113, 7.3858, -5.2017],
                      [3.7843, -8.4365, -4.8026, 0.9972, -6.3218],
                      [-4.9630, -5.1464, 6.0014, -7.1009, -9.8009],
                      [-8.1154, 3.0889, -1.5309, 8.8725, -5.5192],
                      [-9.9770, 4.4352, 5.3100, 3.5803, 3.2389],
                      [5.6948, 4.7685, 0.9319, -7.6121, 6.6583]])

    # Vamos juntar as duas frentes de Pareto em um único arquivo
    Juntos = np.vstack((MOEA1, MOEA2))

    # Agora vamos normalizar o MOEA1 e o MOEA2 entre 0 e 1 em relação ao Máximo e Mínimo da matriz Juntos

    # Calculate the minimum and maximum values for Juntos
    min_Juntos = np.min(Juntos, axis=0)
    max_Juntos = np.max(Juntos, axis=0)

    # Normalize MOEA1
    mi_Pareto_MOEA1 = np.tile(min_Juntos, (MOEA1.shape[0], 1))
    ma_Pareto_MOEA1 = np.tile(max_Juntos, (MOEA1.shape[0], 1))
    MOEA1_Normalizado = (MOEA1 - mi_Pareto_MOEA1) / (ma_Pareto_MOEA1 - mi_Pareto_MOEA1)

    # Normalize MOEA2
    mi_Pareto_MOEA2 = np.tile(min_Juntos, (MOEA2.shape[0], 1))
    ma_Pareto_MOEA2 = np.tile(max_Juntos, (MOEA2.shape[0], 1))
    MOEA2_Normalizado = (MOEA2 - mi_Pareto_MOEA2) / (ma_Pareto_MOEA2 - mi_Pareto_MOEA2)

    # Agora vamos calcular o Hipervolume para o MOEA1 e o MOEA2 normalizados
    HV_MOEA1_Normalizado = wfg(MOEA1_Normalizado, np.ones(5))
    HV_MOEA2_Normalizado = wfg(MOEA2_Normalizado, np.ones(5))
    # print(HV_MOEA1_Normalizado, HV_MOEA2_Normalizado)

    # Os resultados foram muito próximos de 1:
    # HV_MOEA1_Normalizado = 0.999864253859992
    # HV_MOEA2_Normalizado = 0.999999984418132
    # Python: 0.9998642538599926 0.9999999844181313

    # Isso aconteceu pois o ponto de referência foi o máximo de cada função objetivo, ou seja, sem normalizar
    # equivale a todas as entradas iguais a 1000000, o que é muito discrepante
    # em relação aos demais valores entre -10 e 10. Isso joga o valor do
    # hipervolume lá no alto, fazendo com que ele não seja capaz de detectar as diferenças entre
    # os MOEAs.

    # Observe a Figura img\Exemplo na pasta. Se o ponto de referência for muito
    # discrepante em relação à frente de Pareto, o hipervolume fica muito
    # grande independente das soluções na frente de Pareto, tornando o
    # indicador ineficaz.

    # Uma saída é obter o Pareto Combinado e restringir o cálculo
    # às soluções que não ultrapassem o máximo de cada função objetivo definido
    # pelo Pareto Combinado. A ideia é que uma vez que o Pareto Combinado
    # é uma aproximação do Pareto Real, soluções que são discrepantes em
    # relação a ele devem ser ignoradas por não representar de fato uma solução
    # em si do problema. Ou seja, elas não chegam nem perto do Pareto Real e ainda atrapalham o cálculo
    # do HV, portanto devem mesmo ser ignoradas e deletadas do Pareto obtido pelo MOEA.

    # Vamos fazer isso agora: calcular o Pareto Combinado da matriz Juntos
    # Pareto_Combinado = Juntos(paretofront(Juntos),:);
    # Pareto_Combinado = Juntos[pareto.eps_sort(Juntos), :]
    Pareto_Combinado = pareto.eps_sort(Juntos)

    # Vamos pegar o máximo do Pareto_Combinado
    Max_Pareto_Combinado = np.max(Pareto_Combinado, axis=0)
    # Max_Pareto_Combinado = [5.6948    4.7685    6.0014    8.8725    7.0606];
    # [5.6948 4.7685 6.0014 8.8725 7.0606]

    # Vamos restringir as soluções obtidas pelos MOEAs ao máximo do
    # Pareto_Combinado, ignorando e deletando do Pareto obtido pelo MOEA às
    # soluções que ultrapassem o máximo de cada função objetivo definido
    # pelo Pareto Combinado.

    # Assuming MOEA1, MOEA2, and Max_Pareto_Combinado are NumPy arrays
    MOEA1_restrito = []
    for i in range(MOEA1.shape[0]):
        if np.all(MOEA1[i, :] <= Max_Pareto_Combinado):
            MOEA1_restrito.append(MOEA1[i, :])

    MOEA1_restrito = np.array(MOEA1_restrito)  # Convert to a NumPy array

    MOEA2_restrito = []
    for i in range(MOEA2.shape[0]):
        if np.all(MOEA2[i, :] <= Max_Pareto_Combinado):
            MOEA2_restrito.append(MOEA2[i, :])

    MOEA2_restrito = np.array(MOEA2_restrito)  # Convert to a NumPy array

    # Agora as frentes de Pareto obtidas dos MOEAs consideradas são as
    # restritas ao máximo do Pareto combinado:
    # print(MOEA1_restrito)
    # print(MOEA2_restrito)

    # Agora repete-se o procedimento de normalização e cálculo do HV para as
    # matrizes MOEA1_restrito e MOEA2_restrito

    # Assuming MOEA1_restrito, MOEA2_restrito are NumPy arrays

    # Combine MOEA1_restrito and MOEA2_restrito
    Juntos = np.vstack((MOEA1_restrito, MOEA2_restrito))

    # Calculate the minimum and maximum values for Juntos
    min_Juntos = np.min(Juntos, axis=0)
    max_Juntos = np.max(Juntos, axis=0)

    # Normalize MOEA1_restrito
    mi_Pareto_MOEA1_restrito = np.tile(min_Juntos, (MOEA1_restrito.shape[0], 1))
    ma_Pareto_MOEA1_restrito = np.tile(max_Juntos, (MOEA1_restrito.shape[0], 1))
    MOEA1_restrito_Normalizado = (MOEA1_restrito - mi_Pareto_MOEA1_restrito) / (
            ma_Pareto_MOEA1_restrito - mi_Pareto_MOEA1_restrito)

    # Normalize MOEA2_restrito
    mi_Pareto_MOEA2_restrito = np.tile(min_Juntos, (MOEA2_restrito.shape[0], 1))
    ma_Pareto_MOEA2_restrito = np.tile(max_Juntos, (MOEA2_restrito.shape[0], 1))
    MOEA2_restrito_Normalizado = (MOEA2_restrito - mi_Pareto_MOEA2_restrito) / (
            ma_Pareto_MOEA2_restrito - mi_Pareto_MOEA2_restrito)

    # Calculate Hypervolume
    HV_MOEA1_restrito_Normalizado = wfg(MOEA1_restrito_Normalizado, np.ones(5))
    HV_MOEA2_restrito_Normalizado = wfg(MOEA2_restrito_Normalizado, np.ones(5))
    print(HV_MOEA1_restrito_Normalizado, HV_MOEA2_restrito_Normalizado)
    # Os resultados foram:
    # HV_MOEA1_restrito_Normalizado =  0.195045872572915
    # HV_MOEA2_restrito_Normalizado =  0.643220116409763
    # 0.19504587257291467 0.6432201164097627


if __name__ == '__main__':
    main()
