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
import pandas as pd

import Water_treatment_dataset


def optimizeDBScan(X, labels_true, experiment_number, eps, minPtsDBscan, samples, best_rand=0, best_eps=0,
                   best_minpts=0, recursive_counter=0):
    # if recursive_counter >= 1:
    #     DbScan.run(X, labels_true, experiment_number, eps, minPtsDBscan, samples)
    #
    #     return best_rand, best_eps, best_minpts

    # Epsilon
    increase = True
    decrease = False
    epsilonToBeAdjusted = eps
    minPtsToBeAdjusted = minPtsDBscan
    rand_eps_increased = 0
    rand_eps_decreased = 0
    rand_minpts_increased = 0
    rand_minpts_decreased = 0
    best_eps = eps
    best_minpts = minPts

    while increase:

        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, best_minpts, samples)
        except:
            new_rand = rand_eps_increased

        if rand_eps_increased <= new_rand:
            epsilonToBeAdjusted = epsilonToBeAdjusted + 0.01
            if rand_eps_increased < new_rand:
                best_eps = epsilonToBeAdjusted
            rand_eps_increased = new_rand



        else:
            epsilonToBeAdjusted = eps
            increase = False
            decrease = True

    while decrease and epsilonToBeAdjusted > 0:
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, epsilonToBeAdjusted, best_minpts, samples)

        except:
            new_rand = rand_eps_decreased
        if rand_eps_decreased <= new_rand:
            epsilonToBeAdjusted = epsilonToBeAdjusted - 0.01
            if rand_eps_decreased < new_rand and rand_eps_increased < new_rand:
                best_eps = epsilonToBeAdjusted
            rand_eps_decreased = new_rand

        else:
            increase = True
            decrease = False
    # minpts
    while increase:
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, best_eps, minPtsToBeAdjusted,
                                  samples)
        except:
            new_rand = rand_minpts_increased

        if rand_minpts_increased <= new_rand:
            minPtsToBeAdjusted = minPtsToBeAdjusted + 1
            if rand_minpts_increased < new_rand:
                best_minpts = minPtsToBeAdjusted
            rand_minpts_increased = new_rand
        else:
            minPtsToBeAdjusted = minPts
            increase = False
            decrease = True

    while decrease and minPtsToBeAdjusted > 1:
        try:
            new_rand = DbScan.run(X, labels_true, experiment_number, best_eps, minPtsToBeAdjusted,
                                  samples)
        except:
            new_rand = rand_minpts_decreased

        if rand_minpts_decreased <= new_rand:
            minPtsToBeAdjusted = minPtsToBeAdjusted - 1
            if (rand_minpts_decreased < new_rand and rand_minpts_increased < new_rand):
                best_minpts = minPtsToBeAdjusted
            rand_minpts_decreased = new_rand


        else:
            decrease = False

    if rand_eps_decreased < rand_eps_increased:
        best_rand = rand_eps_increased
    if best_rand < rand_minpts_decreased:
        best_rand = rand_minpts_decreased
    if best_rand < rand_minpts_increased:
        best_rand = rand_minpts_increased

    return best_rand, best_eps, best_minpts


def optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue, etavalue, metricvalue, minPts, epsVDBScan,
                    percent_noise_vdbscan, mints_decease_factor_vdbscan, counter=0, best_rand=0, best_kappa=0,
                    best_eta=0):
    rand_kappa_increased = 0
    rand_kappa_decreased = 0
    rand_eta_increased = 0
    rand_eta_decreased = 0
    # increaseKappa
    kappavalueToAdjust = kappavalue
    # print("Kappa initial value: " + str(kappavalue))
    etavalueToAdjust = etavalue
    # print("Eta initial value: " + str(etavalue))
    minptsToAdjust = minPts
    best_eta = etavalue
    best_kappa = kappavalue

    increase = True
    decrease = False

    step_factor = 0.01
    while increase:

        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust,
                                      best_eta, metricvalue, minptsToAdjust, epsVDBScan,
                                      percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            #      print("failed to run VDBSCAN in optimizer")
            new_rand = rand_kappa_increased

        if rand_kappa_increased <= new_rand:
            if rand_kappa_increased < new_rand:
                best_kappa = kappavalueToAdjust
            best_kappa = kappavalueToAdjust
            kappavalueToAdjust = kappavalueToAdjust + step_factor
            rand_kappa_increased = new_rand


        else:
            if step_factor >= 0.001:
                step_factor = step_factor / 10
                kappavalueToAdjust = best_kappa + step_factor

            else:
                kappavalueToAdjust = kappavalue
                increase = False
                decrease = True
                step_factor = 0.01
    while decrease:

        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, kappavalueToAdjust,
                                      best_eta, metricvalue,
                                      minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            #      print("failed to run VDBSCAN in optimizer")
            new_rand = rand_kappa_decreased

        if rand_kappa_decreased <= new_rand and kappavalueToAdjust >= 0:
            #     print("decreasing")

            if rand_kappa_decreased < new_rand and rand_kappa_increased < new_rand:
                best_kappa = kappavalueToAdjust
            best_kappa = kappavalueToAdjust

            kappavalueToAdjust = kappavalueToAdjust - step_factor
            rand_kappa_decreased = new_rand

        else:
            if step_factor >= 0.001:
                step_factor = step_factor / 10
                kappavalueToAdjust = best_kappa - step_factor

            else:
                kappavalueToAdjust = kappavalue

                increase = True
                decrease = False
                step_factor = 0.1

    # EtaOptimize
    # increaseKappa
    # decrease = False

    while increase:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, best_kappa,
                                      etavalueToAdjust, metricvalue, minptsToAdjust, epsVDBScan,
                                      percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            #    print("failed to run VDBSCAN in optimizer")
            new_rand = rand_eta_increased

        if rand_eta_increased <= new_rand:

            if rand_eta_increased < new_rand:
                best_eta = etavalueToAdjust
            best_eta = etavalueToAdjust
            etavalueToAdjust = etavalueToAdjust + step_factor
            rand_eta_increased = new_rand


        else:
            if step_factor >= 0.01:
                step_factor = step_factor / 10
                etavalueToAdjust = best_eta + step_factor

            else:
                etavalueToAdjust = etavalue
                increase = False
                decrease = True
                step_factor = 1

    while decrease and etavalueToAdjust >= 0:
        try:
            new_rand = runVDBScan.run(X, labels_true, experiment_number, samples, best_kappa,
                                      etavalueToAdjust, metricvalue,
                                      minptsToAdjust, epsVDBScan, percent_noise_vdbscan, mints_decease_factor_vdbscan)
        except:
            #   print("failed to run VDBSCAN in optimizer")
            new_rand = rand_eta_decreased

        if rand_eta_decreased <= new_rand:
            if rand_eta_decreased < new_rand and rand_eta_increased < new_rand:
                best_eta = etavalueToAdjust
            best_eta  = etavalueToAdjust
            etavalueToAdjust = etavalueToAdjust - step_factor
            rand_eta_decreased = new_rand

        else:
            if step_factor >= 0.01:
                step_factor = step_factor / 10
                etavalueToAdjust = best_eta - step_factor

            else:
                decrease = False
                # best_eta = etavalueToAdjust
                # etavalueToAdjust = etavalue

    # ÄNDRA
    if rand_kappa_decreased < rand_kappa_increased:
        best_rand = rand_kappa_increased
    if best_rand < rand_eta_increased:
        best_rand = rand_eta_increased
    if best_rand < rand_eta_decreased:
        best_rand = rand_eta_decreased

    return best_rand, best_kappa, best_eta


# Expriment 7 IRIS  ##########################
samples = 250
metricvalue = 'default'
# metricvalue = cosine

kappavalue = 0.005
etavalue = 0.1

K = 3
experiment_number = 7
eps = 0.5
epsVDBScan = 0.7
minPts_modified = (samples / 20) + (0.0001 * samples)
minPts = 3
minPtsDBscan = 3

percent_noise_vdbscan = 20
mints_decease_factor_vdbscan = 0.9

iris = datasets.load_iris()
X = iris.data
labels_true = iris.target
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## IRIS #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## IRIS END #########")

# Expriment Digits  ##########################

digits = datasets.load_digits()
labels_true = digits.target
X = digits.data
X = preprocessing.scale(X)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Digits #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Digits END #########")

# Expriment Diabetes  ##########################


diabetes = datasets.load_diabetes()
X = diabetes.data
labels_true = diabetes.target
X = preprocessing.scale(X)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Diabetes #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Diabetes END #########")

# Expriment Cancer  ##########################

cancer = datasets.load_breast_cancer()
X = cancer.data
labels_true = cancer.target
X = preprocessing.scale(X)
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Cancer #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Cancer END #########")

# Expriment Mall_Customer  ##########################
dataset = pd.read_csv('Mall_Customers.csv')
dataset.isnull().sum()
X= dataset.iloc[:, [3,4]].values
labels_true = dataset.iloc[:, 2].values
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
X =  preprocessing.scale(X)
#Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Mall_customer #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Linnerud END #########")

#Expriment Wine  ##########################

wine = datasets.load_wine()
X = wine.data
labels_true = wine.target
X = preprocessing.scale(X)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Wine #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Wine END #########")

# Expriment Boston  ##########################

boston = datasets.load_boston()
X = boston.data
labels_true = boston.target
X = preprocessing.scale(X)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Run algorithms
best_rand_vdbscan_modified, best_kappa_modified, best_eta_modifed = optimizeVDBScan(X, labels_true, experiment_number,
                                                                                    samples, kappavalue, etavalue,
                                                                                    metricvalue, minPts_modified,
                                                                                    epsVDBScan, percent_noise_vdbscan,
                                                                                    mints_decease_factor_vdbscan)
best_rand_vdbscan, best_kappa, best_eta = optimizeVDBScan(X, labels_true, experiment_number, samples, kappavalue,
                                                          etavalue, metricvalue, minPts, epsVDBScan,
                                                          percent_noise_vdbscan, mints_decease_factor_vdbscan)
best_rand_kmeans = Kmeans.run(X, labels_true, K, experiment_number, samples)
best_rand_dbscan, best_epsilon_dbscan, best_minpts_dbscan = optimizeDBScan(X, labels_true, experiment_number, eps,
                                                                           minPtsDBscan, samples)

print("######## Boston #########")
print("Modified VDBSCAN Optimized rand value: " + str(best_rand_vdbscan_modified))
print("VDBSCAN Optimized rand value: " + str(best_rand_vdbscan))
print("KMEANS Optimized rand value: " + str(best_rand_kmeans))
print("DBSCAN Optimized rand value: " + str(best_rand_dbscan))
print("######## Boston END #########")

