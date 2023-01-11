'''
图像分割: 利用图像的灰度, 颜色, 纹理, 形状等特性, 把图像分成若干个互不重叠的区域, 
并使得这些特征在同一区域内呈现相似性, 在不同区域之间存在明显的差异性. 
然后就可以将分割的图像中具有独特性质的区域提取出来用于不同的研究.

常用分割方法:
    1. 阈值分割: 对图像灰度值进行度量, 设置不同类别的阈值, 达到分割的目的
    2. 边缘分割: 对图像边缘进行检测, 即检测图像中灰度值发生跳变的地方, 则为一片区域的边缘
    3. 直方图法: 对图像的颜色建立直方图, 而直方图的波峰波谷能够表示一块区域的颜色值的范围, 来达到分割的目的.
    4. 基于特定理论: 比如基于聚类分析, 小波变换等理论完成图像分割
'''

# 利用K-means聚类算法对图像像素点颜色进行聚类, 实现简单的图像分割
# 输出: 同一聚类中的点使用相同颜色标记, 不同聚类颜色不同

'''
1. 建立工程
2. 加载图像并进行预处理
3. 加载KMeans聚类算法
4. 对像素进行聚类并输出
'''

from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def loadData(filePath):
    '''
    以二进制的形式读取一张图像
    '''
    f = open(filePath, 'rb')
    img = Image.open(f)
    rows, cols = img.size
    data = []
    for i in range(rows):
        for j in range(cols):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), rows, cols

img_data, rows, cols = loadData('img.jpg')

# KMeans
km = KMeans(n_clusters=3)
label = km.fit_predict(img_data)
label = label.reshape([rows, cols])

pic_new = Image.new("L", (rows, cols))
for i in range(rows):
    for j in range(cols):
        pic_new.putpixel((i, j), int(256 / (label[i][j]+1)))
pic_new.save("result-bull-4.jpg", "JPEG")