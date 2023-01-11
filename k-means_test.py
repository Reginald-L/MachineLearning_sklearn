from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 1. 使用iris数据集
features, labels = load_iris(return_X_y=True)
# 2. 实例化K-means算法
km = KMeans(n_clusters=3)
# 3. 使用fit_predict()方法进行计算
label = km.fit_predict(features)

# 降维
# 构造PCA实例, 并指定维度为2
pca = PCA(n_components=2)
# 进行降维
reduced_features = pca.fit_transform(features)
# 保存
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_features)):
    if labels[i] == 0:
        red_x.append(reduced_features[i][0])
        red_y.append(reduced_features[i][1])
    elif labels[i] == 1:
        blue_x.append(reduced_features[i][0])
        blue_y.append(reduced_features[i][1])
    else:
        green_x.append(reduced_features[i][0])
        green_y.append(reduced_features[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()