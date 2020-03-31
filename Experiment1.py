from sklearn.datasets.samples_generator import make_blobs, make_gaussian_quantiles, make_moons, make_circles;
import runVDBScan
import Kmeans
import DbScan
from scipy.spatial.distance import cosine

import Water_treatment_dataset

# # Data set ##########################
# X, labels_true = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5],
#                             random_state=8)
# K = 2
# experiment_number = 1
# eps = 0.5
# minPts = 3
# # End data set ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number)
# Kmeans.run(X, labels_true, K, experiment_number)
# DbScan.run(X, labels_true, experiment_number, eps, minPts)
# ################
#
#
#
#
# # Data set2 ##########################
# X, labels_true = make_blobs(n_samples=100, cluster_std=[0.5, 2.5, 0.8],
#                               random_state=8)
# K = 3
# experiment_number = 2
# eps = 0.5
# minPts = 3
# # End data set ######################
#
# # Run algorithms2
# runVDBScan.run(X, labels_true, experiment_number)
# Kmeans.run(X, labels_true, K, experiment_number)
# DbScan.run(X, labels_true, experiment_number, eps, minPts)
# ################
#
#
# # Data set3 ##########################
# X, labels_true = make_blobs(n_samples=100, cluster_std=[0.5, 0.4, 0.8],
#                               random_state=8)
# K = 3
# experiment_number = 3
# eps = 0.5
# minPts = 3
# # End data set ######################
#
# # Run algorithms2
# runVDBScan.run(X, labels_true, experiment_number)
# Kmeans.run(X, labels_true, K, experiment_number)
# DbScan.run(X, labels_true, experiment_number, eps, minPts)
# ################
#
#
#
#
# Data set4 ##########################
samples = 1000
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 2
experiment_number = 4
eps = 0.1
minPts = (samples/20) + (0.0001 * samples)

#X, labels_true = make_circles(n_samples=samples, factor=.5, noise=.05)

#X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 2.5, 0.5], random_state=8)
X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
#X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
#X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
#                                 n_samples=500, n_features=2,
#                                 n_classes=2, random_state=1)


#X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# End data set4 ######################

# Run algorithms
runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
#Kmeans.run(X, labels_true, K, experiment_number)
#DbScan.run(X, labels_true, experiment_number, eps, minPts)
################

#
# # # Data set5 ##########################
# # X, labels_true = make_blobs(n_samples=200, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5],
# #                             random_state=8)
# # K = 4
# # experiment_number = 5
# # eps = 3.0
# # minPts = 3
# # # End data set5 ######################
# #
# # # Run algorithms
# # runVDBScan.run(X, labels_true, experiment_number)
# # Kmeans.run(X, labels_true, K, experiment_number)
# # DbScan.run(X, labels_true, experiment_number, eps, minPts)
# # ################
# #
# # Data set5 ##########################
# X = Water_treatment_dataset.run()
# K = 4
# experiment_number = 6
# eps = 3.0
# minPts = 3
# # End data set5 ######################
#
# # Run algorithms
# #runVDBScan.run_without_true_labels(X, experiment_number)
# #Kmeans.run_without_true_labels(X, K, experiment_number)
# runVDBScan.run_without_true_labels(X, experiment_number)
#
# #DbScan_labels = DbScan.run_without_true_labels(X, experiment_number, eps, minPts)
# ################