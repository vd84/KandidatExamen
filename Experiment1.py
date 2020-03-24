

from VDBScan import VDBSCAN
from sklearn.datasets import load_iris
from scipy.spatial.distance import cosine
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import runVDBScan
import Kmeans
import DbScan

#Data set #1#########################
X, labels_true = make_blobs(n_samples=750, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)
K = 2
#End data set #1#####################


#Run algorithms
runVDBScan.run(X, labels_true)
Kmeans.run(X, labels_true, K)
DbScan.run(X, labels_true)
################

