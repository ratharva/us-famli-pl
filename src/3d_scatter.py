import matplotlib.pyplot as plt
import numpy as np

import pickle
import sys
import math

from sklearn.cluster import KMeans


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# lights = pickle.load(open(sys.argv[1], 'rb'))

# print(lights.shape)

# lights = fibonacci_sphere(1000)

lights = np.random.normal(loc=0, scale=1, size=(1000, 3))
lights = np.abs(lights/np.linalg.norm(lights, axis=1, keepdims=True))


# fit KMeans++ model to the data
kmeans = KMeans(n_clusters=32, init='k-means++').fit(lights)

# get the cluster centroids
centroids = kmeans.cluster_centers_


centroids = np.abs(centroids/np.linalg.norm(centroids, axis=1, keepdims=True))


# # calculate the pairwise distances between centroids
# distances = np.sqrt(np.sum((centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=-1))

# # maximize the minimum distance between centroids
# max_distance = np.max(np.min(distances, axis=1))

# ax.scatter(lights[:,0], lights[:,1], lights[:,2], c='red')
ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], linewidths=10)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()