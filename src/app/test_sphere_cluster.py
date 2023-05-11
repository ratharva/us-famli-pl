# Find K clusters from data matrix X (n_examples x n_features)

import numpy as np

K=5

X = np.random.rand(12, 128)

# spherical k-means
# from spherecluster import SphericalKMeans
# skm = SphericalKMeans(n_clusters=K)
# skm.fit(X)

# skm.cluster_centers_
# skm.labels_
# skm.inertia_

# movMF-soft
from spherecluster import VonMisesFisherMixture
vmf_soft = VonMisesFisherMixture(n_clusters=K, posterior_type='soft')
vmf_soft.fit(X)

# vmf_soft.cluster_centers_
# vmf_soft.labels_
# vmf_soft.weights_
# vmf_soft.concentrations_
# vmf_soft.inertia_

# movMF-hard
from spherecluster import VonMisesFisherMixture
vmf_hard = VonMisesFisherMixture(n_clusters=K, posterior_type='hard')
vmf_hard.fit(X)

# vmf_hard.cluster_centers_
# vmf_hard.labels_
# vmf_hard.weights_
# vmf_hard.concentrations_
# vmf_hard.inertia_