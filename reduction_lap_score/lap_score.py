import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W


class LapScore:
    def __init__(self, pop_f):
        # MinMaxScaler show lower changes on the original data
        self.scaled_pop_f = MinMaxScaler().fit_transform(pop_f)

        """
        Construct the affinity matrix W through different ways
        Notes
        -----
        if kwargs is null, use the default parameter settings;
        if kwargs is not null, construct the affinity matrix according to parameters in kwargs
        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        kwargs: {dictionary}
            parameters to construct different affinity matrix W:
            y: {numpy array}, shape (n_samples, 1)
                the true label information needed under the 'supervised' neighbor mode
            metric: {string}
                choices for different distance measures
                'euclidean' - use euclidean distance
                'cosine' - use cosine distance (default)
            neighbor_mode: {string}
                indicates how to construct the graph
                'knn' - put an edge between two nodes if and only if they are among the
                        k nearest neighbors of each other (default)
                'supervised' - put an edge between two nodes if they belong to same class
                        and they are among the k nearest neighbors of each other
            weight_mode: {string}
                indicates how to assign weights for each edge in the graph
                'binary' - 0-1 weighting, every edge receives weight of 1 (default)
                'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                                this weight mode can only be used under 'euclidean' metric and you are required
                                to provide the parameter t
                'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                            this weight mode can only be used under 'cosine' metric
            k: {int}
                choices for the number of neighbors (default k = 5)
            t: {float}
                parameter for the 'heat_kernel' weight_mode
            fisher_score: {boolean}
                indicates whether to build the affinity matrix in a fisher score way, in which W_ij = 1/n_l if yi = yj = l;
                otherwise W_ij = 0 (default fisher_score = false)
            reliefF: {boolean}
                indicates whether to build the affinity matrix in a reliefF way, NH(x) and NM(x,y) denotes a set of
                k nearest points to x with the same class as x, and a different class (the class y), respectively.
                W_ij = 1 if i = j; W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y) (default reliefF = false)
        Output
        ------
        W: {sparse matrix}, shape (n_samples, n_samples)
            output affinity matrix W
        """

        # construct affinity matrix w
        kwargs_w = {"metric": 'euclidean',
                    "neighbor_mode": 'knn',
                    "weight_mode": 'heat_kernel',
                    "k": 5,
                    "t": 1}

        self.w = construct_W.construct_W(self.scaled_pop_f, **kwargs_w)

    def lap_score(self):
        # obtain the scores of features
        score = lap_score.lap_score(self.scaled_pop_f, W=self.w)
        sorted_scores = np.around(np.sort(score, axis=0), decimals=4)

        # sort the feature scores in ascending order according to the feature scores
        idx = lap_score.feature_ranking(score)

        """
        # perform evaluation on clustering task
        number of selected features
        num_cluster = 4  # number of clusters, it is usually set as the number of classes in the ground truth

        # obtain the dataset on the selected features
        selected_features = self.scaled_data[:, idx[0:self.num_fea]]

        print(self.selected_features)
        print(score[idx[0:num_fea]])
        print(selected_features.shape)
        print(idx[0:num_fea])
        """
        return sorted_scores, idx
