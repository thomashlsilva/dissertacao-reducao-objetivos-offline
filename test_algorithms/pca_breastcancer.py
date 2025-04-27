from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    breast = load_breast_cancer()
    breast_data = breast.data
    breast_labels = breast.target
    labels = np.reshape(breast_labels, (569, 1))
    final_breast_data = np.concatenate([breast_data, labels], axis=1)

    breast_dataset = pd.DataFrame(final_breast_data)
    features = breast.feature_names
    features_labels = np.append(features, 'label')
    breast_dataset.columns = features_labels

    breast_dataset['label'].replace(0, 'Benign', inplace=True)
    breast_dataset['label'].replace(1, 'Malignant', inplace=True)

    x = breast_dataset.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features

    feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
    normalised_breast = pd.DataFrame(x, columns=feat_cols)

    pca_breast = PCA(n_components=2)
    principalComponents_breast = pca_breast.fit_transform(x)
    principal_breast_Df = pd.DataFrame(data=principalComponents_breast
                                       , columns=['principal component 1', 'principal component 2'])

    print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("Principal Component Analysis of Breast Cancer Dataset", fontsize=20)
    targets = ['Benign', 'Malignant']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = breast_dataset['label'] == target
        plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
                    , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

    plt.legend(targets, prop={'size': 15})
    plt.show()

if __name__ == '__main__':
    main()
