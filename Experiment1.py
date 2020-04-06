from sklearn.datasets.samples_generator import make_blobs, make_gaussian_quantiles, make_moons, make_circles;
import runVDBScan
import Kmeans
import DbScan
import numpy

from scipy.spatial.distance import cosine

import Water_treatment_dataset


#
# Expriment 1  ##########################
samples = 500
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 1
eps = 0.5
minPts = 5#(samples/20) + (0.0001 * samples)
minPtsDBscan = 5

#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=3)
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
runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
Kmeans.run(X, labels_true, K, experiment_number, samples)
DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
################


#
# Expriment 2  ##########################
samples = 500
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 2
eps = 0.5
minPts = (samples/20) + (0.0001 * samples)
minPtsDBscan = 5

#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=3)
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
runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts)
Kmeans.run(X, labels_true, K, experiment_number, samples)
DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
################