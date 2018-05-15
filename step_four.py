import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from step_one import *
data = new_data
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
labels = kmeans.labels_
my_centroids = kmeans.cluster_centers_
dataset = kmeans.transform(data)
print(labels)
# point_x = dataset[:, 0]
# point_y = dataset[:, 1]
# cent_x = my_centroids[:, 0]
# cent_y = my_centroids[:, 1]
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
# ax.scatter(cent_x, cent_y, s=100, c="black", marker="v", label="centroids")
# ax.legend()
# ax.set_xlabel("factor1")
# ax.set_ylabel("factor2")
# plt.show()

