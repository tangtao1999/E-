import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


data = np.loadtxt('2.txt', delimiter='\t')
scaler = preprocessing.MaxAbsScaler()
data_set = scaler.fit_transform(data)


point_x = data_set[:, 0]
point_y = data_set[:, 1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")
plt.show()
