import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
# import matplotlib.pyplot as plt
pca = PCA(n_components=3)


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
pca.fit(data_set)
# print(pca.explained_variance_ratio_)
new_data = pca.transform(data_set).astype(int)
# print(new_data)
# plt.scatter(new_data[:, 0], new_data[:, 1], marker='o')
# plt.show()


data = pd.DataFrame(data=new_data)
data.to_excel('data.xls')












