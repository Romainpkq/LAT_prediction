# This file is try to use random Forest to reduce the dimensions of features
from random import shuffle

import pandas as pd
import numpy as np
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


from scipy import sparse
from scipy.io import arff


# from .arff get the data
data1, meta = arff.loadarff(open(r'D:\NLP-project\rerealisation\BioMedicalDataSet780s-actual-features.arff',
                                 encoding='utf-8'))
df = pd.DataFrame(data1)
dm = pd.DataFrame(meta)
# print(df.shape)
name1 = (list(df)[85:])


def plot_feature_importances(feature_importances, title, feature_names):
    # sort
    index_sorted = np.flipud(np.argsort(feature_importances))[:20]
    print('index:', len(index_sorted))

    pos = np.arange(index_sorted.shape[0]) + 1

    y_name = [feature_names[i] for i in index_sorted]

    plt.figure(figsize=(128, 32))
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=20)
    plt.xlabel('Features', fontsize=30)
    plt.ylabel('Feature weights', fontsize=30)
    plt.bar(pos, feature_importances[index_sorted], align='center', color='grey')
    plt.xticks(pos, y_name)
    plt.ylabel('Feature weights')
    # plt.title('The weights of the most 20 features(nn experiment)', fontsize=200)
    plt.savefig("examples.png")


def reduce_dimension(data1, label1, dimension_num, estimators=100):
    # The method is to reduce the dimension of vector
    # and choose the most important features

    # print('label1: ', label1.shape)
    y_train = sparse.lil_matrix((label1.shape[0], 85))
    y_train[:, :] = label1
    # print(y_train.shape)

    X_train = sparse.lil_matrix((label1.shape[0], 4189))
    X_train[:, :] = data1
    # print(X_train.shape)

    classifier5 = RandomForestClassifier(n_estimators=estimators, random_state=1)
    classifier1 = LabelPowerset(classifier=classifier5, require_dense=[False, True])
    classifier1.fit(X_train, y_train)

    importances = classifier5.feature_importances_
    # print('importances1: ', importances)
    indices = np.argsort(importances)[::-1]
    # print('indices', indices)
    features_importances = importances[indices]
    # plot_feature_importances(importances, 'Features Importance(Random Forest)', name1)

    return indices[:dimension_num], indices[dimension_num:], features_importances
