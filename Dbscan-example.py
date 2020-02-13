# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:08:08 2020

@author: doha6991
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


from DbScan import MyDBSCAN

# Create three gaussian blobs to use as our clustering data.
centers = [[1, 1], [5, -5], [10, -10]]
X, labels_true = make_blobs(n_samples=750, cluster_std=[1.0, 2.5, 0.5], centers = centers,
                            random_state=8)

X = StandardScaler().fit_transform(X)

###############################################################################
# My implementation of DBSCAN
#

# Run my DBSCAN implementation.
print ('Running my implementation...')
my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)

core_samples_mask = np.zeros_like(my_labels, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True

###############################################################################
# Scikit-learn implementation of DBSCAN
#

print ('Runing scikit-learn implementation...')
db = DBSCAN(eps=0.1, min_samples=3).fit(X)
skl_labels = db.labels_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[my_labels] = True



# Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start
# numbering at 1, so increment the skl cluster numbers by 1.
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1


###############################################################################
# Did we get the same results?

num_disagree = 0

# Go through each label and make sure they match (print the labels if they 
# don't)
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == my_labels[i]:
        print ('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
        num_disagree += 1

if num_disagree == 0:
    print ('PASS - All labels match!')
else:
    print ('FAIL -', num_disagree, 'labels don\'t match.'  )
    
    
    
    
#Plot results of scikit DBSCAN
    
# Black removed and is used for noise instead.
unique_labels = set(skl_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (skl_labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % 750)
plt.show()








