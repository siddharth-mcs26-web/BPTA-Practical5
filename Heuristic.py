import numpy as np
from sklearn.cluster import KMeans

def Heuristic(data_set, k):
    k_mean = KMeans(n_clusters=k)
    k_mean.fit(data_set)

    data_pts = np.array(data_set)
    k_centers = k_mean.cluster_centers_

    closest_points = []
    closest_indices = []

    for center in k_centers:
        
        distances = np.linalg.norm(data_pts - center)

        closest_idx = np.argmin(distances)
        closest_indices.append(closest_idx)
        closest_points.append(data_set[closest_idx])


    min_distance = []

    for i in range(len(data_set)):
        d = []
        for c in closest_indices:
            d.append(np.linalg.norm(data_set[c] - data_set[i]))
        min_distance.append(min(d))

    max_distance = max(min_distance)
    return max_distance
