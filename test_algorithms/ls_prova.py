import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # load data (variables in columns, i.e., reduction in columns)
    # mat = scipy.io.load('COIL20.mat')

    data = np.array([[7587, 321, 112, 950],
                     [6695, 211, 345, 820],
                     [3788, 308, 450, 750],
                     [8108, 278, 88, 999],
                     [5652, 223, 212, 812],
                     [6777, 355, 90, 901],
                     [5812, 401, 185, 788],
                     [7432, 208, 208, 790]])

    # scale features
    scaler = MinMaxScaler()
    model = scaler.fit(data)
    scaled_data = model.transform(data)

    # view normalized data
    print(scaled_data)

    # construct affinity matrix
    kwargs_w = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    w = construct_W.construct_W(scaled_data, **kwargs_w)

    # obtain the scores of features
    score = lap_score.lap_score(scaled_data, W=w)

    # sort the feature scores in ascending order according to the feature scores
    idx = lap_score.feature_ranking(score)

    # perform evaluation on clustering task
    num_fea = 4  # number of selected features
    # num_cluster = 4  # number of clusters, it is usually set as the number of classes in the ground truth

    # obtain the dataset on the selected features
    selected_features = scaled_data[:, idx[0:num_fea]]
    print(selected_features)
    print(score[idx[0:num_fea]])
    print(idx[0:num_fea])
    sns.kdeplot(scaled_data[:, idx[3]])
    sns.rugplot(scaled_data[:, idx[3]])
    plt.show()
    # print(np.size(selected_features, 0))
    # print(np.size(selected_features, 1))
    '''
    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for i in range(0, 20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
        nmi_total += nmi
        acc_total += acc

    # output the average NMI and average ACC
    print('NMI:', float(nmi_total)/20)
    print('ACC:', float(acc_total)/20)
    '''

if __name__ == '__main__':
    main()
