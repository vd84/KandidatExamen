
from sklearn.datasets.samples_generator import make_blobs
import runVDBScan
import Kmeans
import DbScan

#Data set ##########################
X, labels_true = make_blobs(n_samples=50, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)
K = 2
experiment_number = 1
#End data set ######################

#Run algorithms
runVDBScan.run(X, labels_true, experiment_number)
Kmeans.run(X, labels_true, K, experiment_number)
DbScan.run(X, labels_true, experiment_number)
################


#Data set2 ##########################
X2, labels_true2 = make_blobs(n_samples=750, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)
K = 2
experiment_number = 2
#End data set ######################

#Run algorithms2
runVDBScan.run(X2, labels_true2, experiment_number)
Kmeans.run(X2, labels_true2, K, experiment_number)
DbScan.run(X2, labels_true2, experiment_number)
################