# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:33:43 2020

@author: doha6991
"""

from kdbscan import KDBSCAN
from VDBScan import VDBSCAN
from sklearn.datasets import load_iris
from scipy.spatial.distance import cosine

X, labels_true = make_blobs(n_samples=750, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)

X = load_iris().data
alg1 = KDBSCAN(h=0.35,t=0.4,metric=cosine)
alg2 = VDBSCAN(kappa=0.005,metric=cosine)
kde = alg1.fit(X[:,1:3],return_kde = True)
alg2.fit(X,eta=0.5)
alg1_labels = alg1.labels_
alg2_labels = alg2.labels_
KDBSCAN.plot_kdbscan_results(kde)