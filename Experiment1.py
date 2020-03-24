from sklearn.datasets.samples_generator import make_blobs
import runVDBScan
import Kmeans
import DbScan

# Data set ##########################
X, labels_true = make_blobs(n_samples=50, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)
K = 2
experiment_number = 1
eps = 0.5
minPts = 3
# End data set ######################

# Run algorithms
runVDBScan.run(X, labels_true, experiment_number)
Kmeans.run(X, labels_true, K, experiment_number)
DbScan.run(X, labels_true, experiment_number, eps, minPts)
################




# Data set2 ##########################
X, labels_true = make_blobs(n_samples=400, cluster_std=[0.5, 2.5, 0.8],
                              random_state=8)
K = 3
experiment_number = 2
eps = 0.5
minPts = 3
# End data set ######################

# Run algorithms2
runVDBScan.run(X, labels_true, experiment_number)
Kmeans.run(X, labels_true, K, experiment_number)
DbScan.run(X, labels_true, experiment_number, eps, minPts)
################


# Data set2 ##########################
X, labels_true = make_blobs(n_samples=400, cluster_std=[0.5, 0.4, 0.8],
                              random_state=8)
K = 3
experiment_number = 2
eps = 0.5
minPts = 3
# End data set ######################

# Run algorithms2
runVDBScan.run(X, labels_true, experiment_number)
Kmeans.run(X, labels_true, K, experiment_number)
DbScan.run(X, labels_true, experiment_number, eps, minPts)
################
