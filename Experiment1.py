from sklearn.datasets.samples_generator import make_blobs, make_gaussian_quantiles, make_moons, make_circles;
import runVDBScan
import Kmeans
import DbScan
import numpy
from sklearn import datasets
from scipy.spatial.distance import cosine, canberra
from sklearn import decomposition



import Water_treatment_dataset



# # # Expriment 1  ##########################
# # samples = 500
# # metricvalue = 'default'
# # #metricvalue = cosine
# #
# # kappavalue = 0.005
# # etavalue = 0.1
# #
# # K = 3
# # experiment_number = 1
# # eps = 0.5
# # minPts = 5#(samples/20) + (0.0001 * samples)
# # minPtsDBscan = 5
# #
# # #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# # X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=8)
# # #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# # #X = []
# #
# # # labels_true = []
# # # labels_true.extend(labels_true1)
# # # labels_true.extend(labels_true2)
# # # X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# # # X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# # # X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# # #                                 n_samples=500, n_features=2,
# # #                                 n_classes=2, random_state=1)
# # # print(str(labels_true1))
# # # print(X1)
# # # X1 = numpy.array(X1)
# # # X2 = numpy.array(X2)
# # # X = numpy.insert(X1, 1, X2, axis=0)
# #
# # # print("PrintarX")
# # # print(X)
# # # X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # #End data set4 ######################
# #
# # # Run algorithms
# # runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
# # Kmeans.run(X, labels_true, K, experiment_number, samples)
# # DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# # ################
#
#
#
# # Expriment 2  ##########################
# samples = 500
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 2
# eps = 0.5
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=8)
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# #labels_true = []
# #labels_true.extend(labels_true1)
# #labels_true.extend(labels_true2)
# #X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# #X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# #X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# #                                 n_samples=500, n_features=2,
# #                                 n_classes=2, random_state=1)
# #print(str(labels_true1))
# #print(X1)
# #X1 = numpy.array(X1)
# #X2 = numpy.array(X2)
# #X = numpy.insert(X1, 1, X2, axis=0)
#
# #print("PrintarX")
# #print(X)
# #X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # End data set4 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
#

#
# # Expriment 3  ##########################
# samples = 1000
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 3
# eps = 0.5
# minPts = 5 #(samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=8)
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# #labels_true = []
# #labels_true.extend(labels_true1)
# #labels_true.extend(labels_true2)
# #X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# #X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# #X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# #                                 n_samples=500, n_features=2,
# #                                 n_classes=2, random_state=1)
# #print(str(labels_true1))
# #print(X1)
# #X1 = numpy.array(X1)
# #X2 = numpy.array(X2)
# #X = numpy.insert(X1, 1, X2, axis=0)
#
# #print("PrintarX")
# #print(X)
# #X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # End data set4 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
#
#
# # Expriment 4  ##########################
# samples = 1000
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 4
# eps = 0.5
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=8)
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# #labels_true = []
# #labels_true.extend(labels_true1)
# #labels_true.extend(labels_true2)
# #X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# #X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# #X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# #                                 n_samples=500, n_features=2,
# #                                 n_classes=2, random_state=1)
# #print(str(labels_true1))
# #print(X1)
# #X1 = numpy.array(X1)
# #X2 = numpy.array(X2)
# #X = numpy.insert(X1, 1, X2, axis=0)
#
# #print("PrintarX")
# #print(X)
# #X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # End data set4 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################


# Expriment 5  ##########################
samples = 250
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 5
eps = 0.5
epsVDBScan = 1
minPts = 5# (samples/20) + (0.0001 * samples)
minPtsDBscan = 5

percent_noise_vdbscan = 20
mints_decease_factor_vdbscan = 0.9


#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
#X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
#X = []

#labels_true = []
#labels_true.extend(labels_true1)
#labels_true.extend(labels_true2)
#X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
#X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
#X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
#                                 n_samples=500, n_features=2,
#                                 n_classes=2, random_state=1)
#print(str(labels_true1))
#print(X1)
#X1 = numpy.array(X1)
#X2 = numpy.array(X2)
#X = numpy.insert(X1, 1, X2, axis=0)

#print("PrintarX")
#print(X)
#X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# End data set4 ######################

# Run algorithms
runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
Kmeans.run(X, labels_true, K, experiment_number, samples)
DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
###############
#


# Expriment 6  ##########################
samples = 250
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 6
eps = 0.5
epsVDBScan = 1
minPts = (samples/20) + (0.0001 * samples)
minPtsDBscan = 5

percent_noise_vdbscan = 20
mints_decease_factor_vdbscan = 0.9


#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
#X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
#X = []

#labels_true = []
#labels_true.extend(labels_true1)
#labels_true.extend(labels_true2)
#X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
#X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
#X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
#                                 n_samples=500, n_features=2,
#                                 n_classes=2, random_state=1)
#print(str(labels_true1))
#print(X1)
#X1 = numpy.array(X1)
#X2 = numpy.array(X2)
#X = numpy.insert(X1, 1, X2, axis=0)

#print("PrintarX")
#print(X)
#X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# End data set4 ######################

# Run algorithms
runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
Kmeans.run(X, labels_true, K, experiment_number, samples)
DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
###############










# # Expriment 7  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 7
# eps = 0.5
# epsVDBScan = 1
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# #X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# iris = datasets.load_iris()
# X = iris.data
# labels_true = iris.target
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# #labels_true = []
# #labels_true.extend(labels_true1)
# #labels_true.extend(labels_true2)
# #X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# #X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# #X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# #                                 n_samples=500, n_features=2,
# #                                 n_classes=2, random_state=1)
# #print(str(labels_true1))
# #print(X1)
# #X1 = numpy.array(X1)
# #X2 = numpy.array(X2)
# #X = numpy.insert(X1, 1, X2, axis=0)
#
# #print("PrintarX")
# #print(X)
# #X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # End data set7 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################

# # Expriment 8  ##########################
# samples = 300
# metricvalue = 'default'
# #metricvalue = cosine
# #metricvalue = canberra
#
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 8
# eps = 0.3
# epsVDBScan = 0.5
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 100
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X1, labels_true1 = make_blobs(n_samples=samples, cluster_std=[0.5, 1.0, 0.1], random_state=3)
# X2, labels_true2 = make_moons(n_samples=300, noise=.05, random_state=5)
# #X3, labels_true3 = make_circles(n_samples=30, factor=.7, noise=.05,)
#
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# labels_true = []
# labels_true.extend(labels_true1)
# labels_true.extend(labels_true2)
# #labels_true.extend(labels_true3)
#
#
# X1 = numpy.array(X1)
# X2 = numpy.array(X2)
# #X3 = numpy.array(X3)
# X = numpy.insert(X1, 1, X2, axis=0)
# #X = numpy.insert(X, 1, X3, axis=0)
#
# # End data set8 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
#
#
# # Expriment 9  ##########################
# samples = 300
# metricvalue = 'default'
# #metricvalue = cosine
# metricvalue = canberra
#
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 9
# eps = 0.3
# epsVDBScan = 0.5
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 100
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X1, labels_true1 = make_blobs(n_samples=samples, cluster_std=[0.5, 1.0, 0.1], random_state=3)
# X2, labels_true2 = make_moons(n_samples=300, noise=.05, random_state=5)
# #X3, labels_true3 = make_circles(n_samples=30, factor=.7, noise=.05,)
#
# #X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# #X = []
#
# labels_true = []
# labels_true.extend(labels_true1)
# labels_true.extend(labels_true2)
# #labels_true.extend(labels_true3)
#
#
# X1 = numpy.array(X1)
# X2 = numpy.array(X2)
# #X3 = numpy.array(X3)
# X = numpy.insert(X1, 1, X2, axis=0)
# #X = numpy.insert(X, 1, X3, axis=0)
#
# # End data set9 ######################
#
# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################