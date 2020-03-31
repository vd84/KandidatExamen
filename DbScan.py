# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:08:08 2020

@author: doha6991
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score

from MyDbScan import MyDBSCAN


# Create three gaussian blobs to use as our clustering data.


def run(X, labels_true, experiment_number, eps, minPts, samples):
    db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
    skl_labels = db.labels_

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    # Plot results of scikit DBSCAN

    # Black removed and is used for noise instead.
    unique_labels = set(skl_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (skl_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plot_text = "DBSCAN PLOT, experiment number: " + str(experiment_number)  + "Samplesize = " + samples
    plt.title(plot_text)
    plt.show()

    silhouette_avg = silhouette_score(X, skl_labels)
    print("DBSCAN: ", "Experiment number ", experiment_number, " The average silhouette_score is :", silhouette_avg)

    rand_score = adjusted_rand_score(labels_true, skl_labels)
    print("DBSCAN: ", "Experiment number ", experiment_number, " The rand index is :", rand_score)
def run_without_true_labels(X, experiment_number, eps, minPts, samples):
    db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
    skl_labels = db.labels_

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    # Plot results of scikit DBSCAN

    # Black removed and is used for noise instead.
    unique_labels = set(skl_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (skl_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plot_text = "DBSCAN PLOT, experiment number: " + str(experiment_number) + "Samplesize = " + samples
    plt.title(plot_text)
    plt.show()

    silhouette_avg = silhouette_score(X, skl_labels)
    print("DBSCAN: ", "Experiment number ", experiment_number, " The average silhouette_score is :", silhouette_avg)
