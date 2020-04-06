from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score

def run(X, labels_true, number_of_clusters, experiment_number, samples):

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(X)

    kmeans_labels = kmeans.labels_

    core_samples_mask = np.zeros_like(kmeans.labels_, dtype=bool)

    unique_labels = set(kmeans_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (kmeans_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plot_text = "KMEANS PLOT, experiment number: " + str(experiment_number) + " Samplesize = " + str(samples) + "\n" + " Clusters found = " + str(number_of_clusters) + "\n"
    plt.title(plot_text)
    plt.show()

    silhouette_avg = silhouette_score(X, kmeans_labels)
    print("KMEANS: ", "Experiment number ", experiment_number," For n_clusters =", number_of_clusters,
              "KMEANS: The average silhouette_score is :", silhouette_avg)

    rand_score = adjusted_rand_score(labels_true, kmeans_labels)
    print("KMEANS: ", "Experiment number ", experiment_number," For n_clusters =", number_of_clusters,
              "KMEANS: The rand index is :", rand_score)

def run_without_true_labels(X, number_of_clusters, experiment_number, samples):

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(X)

    kmeans_labels = kmeans.labels_

    core_samples_mask = np.zeros_like(kmeans.labels_, dtype=bool)

    unique_labels = set(kmeans_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (kmeans_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plot_text = "KMEANS PLOT, experiment number: " + str(experiment_number)  + "Samplesize = " + samples + "k= " + str(number_of_clusters)
    plt.title(plot_text)
    plt.show()

    silhouette_avg = silhouette_score(X, kmeans_labels)
    print("KMEANS: ", "Experiment number ", experiment_number," For n_clusters =", number_of_clusters,
              "KMEANS: The average silhouette_score is :", silhouette_avg)

