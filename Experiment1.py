from sklearn.datasets.samples_generator import make_blobs, make_gaussian_quantiles, make_moons, make_circles;
import runVDBScan
import Kmeans
import DbScan
import numpy
from sklearn import datasets
from scipy.spatial.distance import cosine, canberra
from sklearn import decomposition
from sklearn import preprocessing
from random import randrange


import Water_treatment_dataset


def optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples, best_rand=0, best_eps = 0, best_minpts = 0, recursive_counter=0):
    if recursive_counter >= 100:
        DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)

        return best_rand, best_eps, best_minpts

    # Epsilon
    increase = True
    decrease = False
    epsilonToBeAdjusted = eps
    minPtsToBeAdjusted = minPtsDBscan
    rand = best_rand
    best_eps = best_eps
    best_minpts = best_minpts


    while increase:
        print("increase")

        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted + 0.01, minPtsToBeAdjusted, samples)
        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand

        if rand < new_rand:
            rand = new_rand
            epsilonToBeAdjusted = epsilonToBeAdjusted + 0.01
            best_eps = epsilonToBeAdjusted

        else:
            increase = False
            decrease = True

    while decrease:
        print("deacrese")
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted - 0.01, minPtsToBeAdjusted, samples)

        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            epsilonToBeAdjusted = epsilonToBeAdjusted - 0.01
            best_eps = epsilonToBeAdjusted
        else:
            decrease = False
            increase = True


    # minpts
    increase = True
    decrease = False
    while increase:
        print("increas2")
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted + 1,
                                  samples)
        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand

        if rand < new_rand:
            rand = new_rand
            minPtsToBeAdjusted = minPtsToBeAdjusted + 1
            best_minpts = minPtsToBeAdjusted


        else:
            increase = False
            decrease = True

    while decrease:
        print("decrease2")
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted - 1,
                                  samples)

        except:
            print("failed to run DBSCAN in optimizer")
            new_rand = rand
        if rand < new_rand:
            rand = new_rand
            if minPtsToBeAdjusted >= 2:
                minPtsToBeAdjusted = minPtsToBeAdjusted - 1
                best_minpts = minPtsToBeAdjusted

        else:
            decrease = False

    return optimizeDBScan(X, labels_true, experiment_number, epsilonToBeAdjusted, minPtsToBeAdjusted, samples, rand, best_eps, best_minpts,
                          recursive_counter + 1)


def optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan,
                    percent_noise_vdbscan, mints_decease_factor_vdbscan, counter=0, best_rand=0, best_kappa = 0, best_eta = 0):
    if counter >= 20:
        return best_rand, best_kappa, best_eta  # KappaOptimize
    rand = best_rand

    iteration_vdbscan_optimizer = 0
    # increaseKappa
    kappavalueToAdjust = kappavalue
    max_iter = 50
    print("Kappa initial value: " + str(kappavalue))
    etavalueToAdjust = etavalue
    print("Eta initial value: " + str(etavalue))
    minptsToAdjust = minPts
    best_eta = best_eta
    best_kappa = best_kappa

    increase = True
    decrease = False
    while increase and not iteration_vdbscan_optimizer >= max_iter:

        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
                                      round(etavalueToAdjust, 1), metricvalue, minptsToAdjust, epsVDBScan,
                                      percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
            etavalueToAdjust = etavalueToAdjust*randrange(2)
            kappavalueToAdjust = kappavalueToAdjust*randrange(2)


        if rand < new_rand:
            rand = new_rand
            kappavalueToAdjust = kappavalueToAdjust + 0.001
            iteration_vdbscan_optimizer = iteration_vdbscan_optimizer + 1
            best_kappa = kappavalueToAdjust
        else:
            increase = False
            decrease = True
    iteration_vdbscan_optimizer = 0
    while decrease and not iteration_vdbscan_optimizer >= max_iter:

        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
                                      round(etavalueToAdjust, 1), metricvalue,
                                      minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
            etavalueToAdjust = etavalueToAdjust*randrange(2)
            kappavalueToAdjust = kappavalueToAdjust*randrange(2)

        if rand < new_rand:
            print("decreasing")
            rand = new_rand
            kappavalueToAdjust = kappavalueToAdjust - 0.001
            iteration_vdbscan_optimizer = iteration_vdbscan_optimizer + 1
            best_kappa = kappavalueToAdjust

        else:
            kappavalueToAdjust = kappavalueToAdjust + 0.001

            decrease = False
    iteration_vdbscan_optimizer = 0

    # EtaOptimize
    # increaseKappa
    increase = True
    decrease = False

    while increase and not iteration_vdbscan_optimizer >= max_iter:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
                                      round(etavalueToAdjust, 1), metricvalue, minptsToAdjust, epsVDBScan,
                                      percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
            etavalueToAdjust = etavalueToAdjust*randrange(2)
            kappavalueToAdjust = kappavalueToAdjust*randrange(2)

        if rand < new_rand:
            rand = new_rand
            etavalueToAdjust = etavalueToAdjust + 0.1
            iteration_vdbscan_optimizer = iteration_vdbscan_optimizer + 1
            best_eta = etavalueToAdjust



        else:
            etavalueToAdjust = etavalue
            increase = False
            decrease = True
    iteration_vdbscan_optimizer = 0

    while decrease and not iteration_vdbscan_optimizer >= max_iter:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
                                      round(etavalueToAdjust, 1), metricvalue,
                                      minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            print("failed to run VDBSCAN in optimizer")
            new_rand = rand
            etavalueToAdjust = etavalueToAdjust*randrange(2)
            kappavalueToAdjust = kappavalueToAdjust*randrange(2)



        if rand < new_rand:
            rand = new_rand
            etavalueToAdjust = etavalueToAdjust - 0.1
            iteration_vdbscan_optimizer = iteration_vdbscan_optimizer + 1
            best_eta = etavalueToAdjust


        else:
            etavalueToAdjust = etavalueToAdjust + 0.1

            decrease = False

    # # Minpts Optimize
    # increase = True
    # decrease = False
    # while increase  and not iteration_vdbscan_optimizer >= max_iter:
    #
    #     try:
    #         new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
    #                                   round(etavalueToAdjust, 1), metricvalue,
    #                                   minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
    #     except:
    #         print("failed to run VDBSCAN in optimizer")
    #         new_rand = rand
    #
    #     if rand < new_rand or (rand == new_rand and rand != 0.0):
    #         rand = new_rand
    #         minptsToAdjust = minptsToAdjust + 1
    #         iteration_vdbscan_optimizer = iteration_vdbscan_optimizer + 1
    #     else:
    #         increase = False
    #         decrease = True
    #         iteration_vdbscan_optimizer = 0
    # while decrease and iteration_vdbscan_optimizer <= max_iter:
    #     try:
    #         new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
    #                                   round(etavalueToAdjust, 1), metricvalue,
    #                                   minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
    #     except:
    #         print("failed to run VDBSCAN in optimizer")
    #         new_rand = rand
    #     if rand < new_rand or (rand == new_rand and rand != 0.0):
    #         rand = new_rand
    #         minptsToAdjust = minptsToAdjust - 1
    #     else:
    #         minptsToAdjust = minptsToAdjust + 1
    #
    #         decrease = False

    return optimizeVDBScan(X, labels_true, experiment_number, samples, round(kappavalueToAdjust, 3),
                           round(etavalueToAdjust, 1), metricvalue, minptsToAdjust, epsVDBScan, percent_noise_vdbscan,
                           mints_decease_factor_vdbscan, counter + 1, rand, best_kappa, best_eta)


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

# #REAL WORLD DATA SETS################
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
# pca = decomposition.PCA(n_components=2)
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
#
# # Run algorithms
# best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
# best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
#
#
# print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
# print("VDBSCAN Optimized kappa value: " + str(best_kappa))
# print("VDBSCAN Optimized eta value: " + str(best_eta))
#
#
# print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
#
# print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
# print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
# print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))
#
#
# ################
# ################


# # Expriment 8  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 8
# eps = 0.5
# epsVDBScan = 0.25
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 100
# mints_decease_factor_vdbscan = 0.9
#
#
#
#
# #X1, abels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# #X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# digits = datasets.load_digits()
# labels_true = digits.target
# X = digits.data
# X = preprocessing.scale(X)
#
# pca = decomposition.PCA(n_components=2)
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
# # End data set8 ######################
#
# # Run algorithms
# # Ingen mod eller optimering VDBSCAN
# # best_rand1, best_kappa1, best_eta1 = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, 5, 0.2, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #
# #
# # print("Optimized rand value: " + str(best_rand1))
# # print("Optimized kappa value: " + str(best_kappa1))
# # print("Optimized eta value: " + str(best_eta1))
# #
#
#
#
#
# best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
#
#
# print("Optimized rand value: " + str(best_rand_dbscan))
# print("Optimized eps value: " + str(best_epsilon_dbscan))
# print("Optimized minpts value: " + str(best_minpts_dbscan))
#
# print("TESTING VALUES")
# DbScan.run(X, labels_true, experiment_number, best_epsilon_dbscan, best_minpts_dbscan, samples)
# #runVDBScan.run(X, labels_true, experiment_number, samples, 0.005, 0.1, metricvalue, minPts, 0.2, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #minPts = (samples/20) + (0.0001 * samples)
# #runVDBScan.run(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #Kmeans.run(X, labels_true, K, experiment_number, samples)
# #DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
# ################
# ################


# # Expriment 9  ##########################
# samples = 250
# metricvalue = 'default'
# # metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.01
#
# K = 3
# experiment_number = 9
# eps = 2
# epsVDBScan = 0.2
# minPts = (samples/20) + (0.0001 * samples)
# minPtsDBscan = 5
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
# # X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# # X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# diabetes = datasets.load_breast_cancer()
# X = diabetes.data
# labels_true = diabetes.target
# X = preprocessing.scale(X)
#
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
#
# # X2, labels_true2 = make_blobs(n_samples=samples, cluster_std=[1.0, 1.5, 0.5], random_state=2)
# # X = []
#
# # labels_true = []
# # labels_true.extend(labels_true1)
# # labels_true.extend(labels_true2)
# # X, labels_true = make_blobs(n_samples=samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=8)
# # X, labels_true = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=8, cov=5)
# # X, labels_true = make_gaussian_quantiles(mean=(4, 4), cov=1,
# #                                 n_samples=500, n_features=2,
# #                                 n_classes=2, random_state=1)
# # print(str(labels_true1))
# # print(X1)
# # X1 = numpy.array(X1)
# # X2 = numpy.array(X2)
# # X = numpy.insert(X1, 1, X2, axis=0)
#
# # print("PrintarX")
# # print(X)
# # X, labels_true = make_moons(n_samples=samples, noise=.05, random_state=1)
# # End data set7 ######################
#
# # Run algorithms
# best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
#                                                           etavalue, metricvalue, minPts, epsVDBScan,
#                                                           percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
# #best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
#                                                                           # minPtsDBscan, samples)
#
# print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
# print("VDBSCAN Optimized kappa value: " + str(best_kappa))
# print("VDBSCAN Optimized eta value: " + str(best_eta))
#
# # print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
# #
# # print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
# # print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
# # print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))

################
################

# # Expriment 10  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 10
# eps = 1
# epsVDBScan = 0.2
# minPts = 3#(samples/20) + (0.0001 * samples)
# minPtsDBscan = 3
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# #X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# cancer = datasets.load_breast_cancer()
# X = cancer.data
# labels_true = cancer.target
# X = preprocessing.scale(X)
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
#
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
#
# # Run algorithms
# best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
# #best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
#
#
# print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
# print("VDBSCAN Optimized kappa value: " + str(best_kappa))
# print("VDBSCAN Optimized eta value: " + str(best_eta))
#
#
# # print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
# #
# # print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
# # print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
# # print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))
#
#
# ################
# ################


# # Expriment 11  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 11
# eps = 1
# epsVDBScan = 0.2
# minPts = 3#(samples/20) + (0.0001 * samples)
# minPtsDBscan = 3
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# #X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# linnerud = datasets.load_linnerud()
# X = linnerud.data
# labels_true = linnerud.target
#
#
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
#
# # Run algorithms
# best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
# #best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
#
#
# print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
# print("VDBSCAN Optimized kappa value: " + str(best_kappa))
# print("VDBSCAN Optimized eta value: " + str(best_eta))
#
#
# # print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
# #
# # print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
# # print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
# # print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))


################
################


# # Expriment 12  ##########################
# samples = 250
# metricvalue = 'default'
# #metricvalue = cosine
#
# kappavalue = 0.005
# etavalue = 0.1
#
# K = 3
# experiment_number = 12
# eps = 1
# epsVDBScan = 0.2
# minPts = 3#(samples/20) + (0.0001 * samples)
# minPtsDBscan = 3
#
# percent_noise_vdbscan = 20
# mints_decease_factor_vdbscan = 0.9
#
#
# #X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
# #X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
# wine = datasets.load_wine()
# X = wine.data
# labels_true = wine.target
# X = preprocessing.scale(X)
#
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
#
#
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
#
# # Run algorithms
# best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
# #best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
# #best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
#
#
# print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
# print("VDBSCAN Optimized kappa value: " + str(best_kappa))
# print("VDBSCAN Optimized eta value: " + str(best_eta))
#
#
# # print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
# #
# # print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
# # print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
# # print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))
#
#
# ################
# ################


# Expriment 13  ##########################
samples = 250
metricvalue = 'default'
#metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 13
eps = 1
epsVDBScan = 0.2
minPts = (samples/20) + (0.0001 * samples)
minPtsDBscan = 3

percent_noise_vdbscan = 20
mints_decease_factor_vdbscan = 0.9


#X1, labels_true1 = make_circles(n_samples=samples, factor=.5, noise=.05)
#X, labels_true = make_blobs(n_samples=samples, cluster_std=[1.8, 2.8, 0.3], random_state=3)
boston = datasets.load_boston()
X = boston.data
labels_true = boston.target
X = preprocessing.scale(X)

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
# End data set7 ######################

# Run algorithms
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples)


print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("VDBSCAN Optimized kappa value: " + str(best_kappa))
print("VDBSCAN Optimized eta value: " + str(best_eta))


print("KMEANS Optimized rand value: " + str(best_rand_kmeans))

print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("DBSCAN Optimized eps value: " + str(best_epsilon_dbscan))
print("DBSCAN Optimized minpts value: " + str(best_minpts_dbscan))


################
################


