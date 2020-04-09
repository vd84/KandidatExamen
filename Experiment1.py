from sklearn.datasets.samples_generator import make_blobs, make_gaussian_quantiles, make_moons, make_circles;
import runVDBScan
import Kmeans
import DbScan
import numpy
from sklearn import datasets
from scipy.spatial.distance import cosine, canberra
from sklearn import decomposition
from sklearn import preprocessing




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
# epsVDBScan = 0.2
#
#
# percent_noise_vdbscan = 100
# mints_decease_factor_vdbscan = 0.9
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
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
# #
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
# epsVDBScan = 0.2
#
#
# percent_noise_vdbscan = 100
# mints_decease_factor_vdbscan = 0.9
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
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################


# # Expriment 5  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 5
# eps = 0.5
# epsVDBScan = 1
# minPts = 5# (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
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
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ###############
# #


# # Expriment 6  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 6
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
# X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
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
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ###############
#
#
#
#

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
#
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

# # Run algorithms
# runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# Kmeans.run(X, labels_true, K, experiment_number, samples)
# DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
# ################



def optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples, best_rand = 0, recursive_counter = 0):
    if recursive_counter >= 3:
        DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)

        return best_rand, epsVDBScan, minPtsDBscan

    #Epsilon
    increase = True
    decrease = False
    epsilonToBeAdjusted = eps
    minPtsToBeAdjusted = minPtsDBscan
    rand = best_rand
    while increase:
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted, samples)
        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand

        if rand < new_rand:
            rand = new_rand
            epsilonToBeAdjusted = epsilonToBeAdjusted + 0.01

        else:
            increase = False
            decrease = True

    while decrease:
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted, samples)

        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            epsilonToBeAdjusted = epsilonToBeAdjusted - 0.01
        else:
            epsilonToBeAdjusted = epsilonToBeAdjusted + 0.01

        # minpts
        increase = True
        decrease = False
        while increase:
            try:
                new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted,
                                      samples)
            except:
                print("failed to run DBSCAN in optimizer")
                new_rand = rand

            if rand < new_rand:
                rand = new_rand
                minPtsToBeAdjusted = minPtsToBeAdjusted + 1

            else:
                increase = False
                decrease = True

        while decrease:
            try:
                new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted,
                                      samples)

            except:
                print("failed to run DBSCAN in optimizer")
                new_rand = rand
            if rand < new_rand:
                rand = new_rand
                minPtsToBeAdjusted = minPtsToBeAdjusted - 1
            else:
                minPtsToBeAdjusted = minPtsToBeAdjusted + 1
                decrease = False
        return optimizeDBScan(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted, samples, rand, recursive_counter + 1)





def optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan, counter = 0):
    if counter >= 100:
        rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue,
                       minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        return rand, kappavalue, etavalue    #KappaOptimize
    rand = 0
    #increaseKappa
    kappavalueToAdjust = kappavalue
    print("Kappa initial value: " + str(kappavalue))
    etavalueToAdjust = etavalue
    print("Eta initial value: " + str(etavalue))

    increase = True
    decrease = False
    while increase:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand

        if rand < new_rand:
            rand = new_rand
            kappavalueToAdjust = kappavalueToAdjust + 0.01
        else:
            increase = False
            decrease = True
    while decrease:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalue, metricvalue,
                                      minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            kappavalueToAdjust = kappavalueToAdjust - 0.01
        else:
            kappavalueToAdjust = kappavalueToAdjust + 0.01

            decrease = False


    #EtaOptimize
    #increaseKappa
    increase = True
    decrease = False

    while increase:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalueToAdjust, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            etavalueToAdjust = etavalueToAdjust + 0.1

        else:
            etavalueToAdjust = etavalue
            increase = False
            decrease = True

    while decrease:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalueToAdjust, metricvalue,
                                      minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            etavalueToAdjust = etavalueToAdjust - 0.1
        else:
            etavalueToAdjust = etavalueToAdjust + 0.1

            decrease = False
    runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalueToAdjust, metricvalue,
                                      minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
    return optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalueToAdjust, etavalueToAdjust, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan, counter + 1)




# Expriment 8  ##########################
samples = 250
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 8
eps = 1
epsVDBScan = 0.25
minPts = (samples/20) + (0.0001 * samples)
minPtsDBscan = 2

percent_noise_vdbscan = 100
mints_decease_factor_vdbscan = 0.9




#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
#X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
X, labels_true = datasets.load_digits(return_X_y=True)
print(X)
#X = preprocessing.scale(X)

# X = dataset.data
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
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
# End data set8 ######################

# Run algorithms
# Ingen mod eller optimering VDBSCAN
# best_rand1, best_kappa1, best_eta1 = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, 5, 0.2, percent_noise_vdbscan, mints_decease_factor_vdbscan)
#
#
# best_rand2, best_kappa2, best_eta2 = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
#
# print("Optimized rand value: " + str(best_rand1))
# print("Optimized kappa value: " + str(best_kappa1))
# print("Optimized eta value: " + str(best_eta1))
#
#
# print("Optimized rand value: " + str(best_rand2))
# print("Optimized kappa value: " + str(best_kappa2))
# print("Optimized eta value: " + str(best_eta2))
#


best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)


print("Optimized rand value: " + str(best_rand_dbscan))
print("Optimized eps value: " + str(best_epsilon_dbscan))
print("Optimized minpts value: " + str(best_minpts_dbscan))



#runVDBScan.run(X, labels_true, experiment_number, samples, 0.005, 0.1, metricvalue, minPts, 0.2, percent_noise_vdbscan, mints_decease_factor_vdbscan)
#minPts = (samples/20) + (0.0001 * samples)
#runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
#Kmeans.run(X, labels_true, K, experiment_number, samples)
#DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
################
################





