# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:33:43 2020

@author: doha6991
"""

from kdbscan import KDBSCAN
from VDBScan import VDBSCAN
from sklearn.datasets import load_iris
from scipy.spatial.distance import cosine
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score


def run(X, labels_true):

    alg2 = VDBSCAN(kappa=0.005,metric=cosine)
    alg2.fit(X,eta=0.5)
    alg2_labels = alg2.labels_

    core_samples_mask = np.zeros_like(alg2_labels, dtype=bool)

    unique_labels = set(alg2_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (alg2_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % 750)
    plt.show()

    silhouette_avg = silhouette_score(X, alg2_labels)
    print("VDBSCAN: The average silhouette_score is :", silhouette_avg)

    rand_score = adjusted_rand_score(labels_true, alg2_labels)
    print("VBSCAN: The rand index is :", rand_score)



