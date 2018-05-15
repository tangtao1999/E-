import numpy as np
import matplotlib.pyplot as plt
from step_one import *


def load_dataset(file_name):
    data_mat = []
    with open(file_name, encoding='gb18030', errors='ignore') as fr:
        lines = fr.readlines()
    for line in lines:
        cur_line = line.strip().split("\t")
        flt_line = list(map(lambda x: float(x), cur_line))
        data_mat.append(flt_line)
    return np.array(data_mat)


data_set = new_data


def dist_eclud(vecA, vecB):
    vec_square = []
    for element in vecA - vecB:
        element = element ** 2
        vec_square.append(element)
    return sum(vec_square) ** 0.5


def rand_cent(data_set, k):
    n = data_set.shape[1]
    centroids = np.zeros((k, n))
    for j in range(n):
        min_j = float(min(data_set[:, j]))
        range_j = float(max(data_set[:, j])) - min_j
        centroids[:, j] = (min_j + range_j * np.random.rand(k, 1))[:, 0]
    return centroids


def Kmeans(data_set, k):
    m = data_set.shape[0]
    cluster_assment = np.zeros((m, 2))
    centroids = rand_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf; min_index = -1
            for j in range(k):
                dist_ji = dist_eclud(centroids[j, :], data_set[i,:])
                if dist_ji < min_dist:
                    min_dist = dist_ji; min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist**2
        for cent in range(k):
            pts_inclust = data_set[np.nonzero(list(map(lambda x:x == cent, cluster_assment[:, 0])))]
            centroids[cent, :] = np.mean(pts_inclust, axis=0)
    return centroids, cluster_assment


my_centroids, my_cluster_assment = Kmeans(data_set, 4)
print(my_centroids)

point_x = data_set[:, 0]
point_y = data_set[:, 1]
cent_x = my_centroids[:, 0]
cent_y = my_centroids[:, 1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
ax.scatter(cent_x, cent_y, s=100, c="black", marker="v", label="centroids")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")
plt.show()
