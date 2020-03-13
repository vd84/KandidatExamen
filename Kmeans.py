from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

number_of_clusters = 3

X, labels_true = make_blobs(n_samples=750, cluster_std=[1.0, 2.5, 0.5],
                            random_state=8)
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

plt.title('Estimated number of clusters: %d' % 750)
plt.show()

silhouette_avg = silhouette_score(X, kmeans_labels)
print("For n_clusters =", number_of_clusters,
          "The average silhouette_score is :", silhouette_avg)