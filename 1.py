import numpy as np
import math
import matplotlib.pyplot as plt


def load_dataset(file_name):
    data_mat = []
    with open(file_name, encoding='gb18030', errors='ignore') as fr:
        lines = fr.readlines()
    for line in lines:
        cur_line = line.strip().split("\t")
        flt_line = list(map(lambda x: float(x), cur_line))
        data_mat.append(flt_line)
    return np.array(data_mat)


data_set = load_dataset(r"1.txt")
print(data_set)

point_x = data_set[:, 0]
point_y = data_set[:, 1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")
plt.show()
