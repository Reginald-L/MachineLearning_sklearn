from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
from numpy.random import RandomState

# 加载人脸数据集
face_data = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
# 展示6张图像
print()
fig = plt.figure(figsize=(15, 8), dpi=80)
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    ax.imshow(face_data.data[i].reshape((64, 64)), cmap='gray')

plt.show()

# 实例化NMF
nmf = NMF(n_components=6)
nmf.fit(face_data.data)
components_ = nmf.components_

fig = plt.figure(figsize=(15, 8), dpi=80)
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    ax.imshow(components_[i].reshape((64, 64)), cmap='gray')

plt.show()