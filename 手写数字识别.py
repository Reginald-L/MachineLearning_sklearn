from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import random

# 1. 加载手写数字数据集
# X.shape = (1797, 64)
# y.shape = 1797
X, y = datasets.load_digits(return_X_y=True)

# 2. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3. 构造KNN算法
"""
n=2: 精度=98.44%
n=3: 精度=98.67%
n=5: 精度=98%
n=6: 精度=97.56%
n=7: 精度=97.78%
n=10: 精度=97.56%
"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
true_num = (y_predict.astype(y_test.dtype) == y_test).sum()
knn_accuracy = true_num / len(y_test)
print(f'KNN精度: {knn_accuracy * 100} %')

# 4. 决策树
"""
gini: 精度: 85.33 %
entropy: 精度: 84.0 %
"""
dec_tree = DecisionTreeClassifier(criterion='gini')
dec_tree.fit(X_train, y_train)
y_predict = dec_tree.predict(X_test)
true_num = (y_predict.astype(y_test.dtype) == y_test).sum()
dt_accuracy = true_num / len(y_test)
print(f'决策树精度: {dt_accuracy * 100} %')

# 5. 随机森林
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)
true_num = (y_predict.astype(y_test.dtype) == y_test).sum()
dt_accuracy = true_num / len(y_test)
print(f'随机森林精度: {dt_accuracy * 100} %')

# 6. 高斯贝叶斯
"""
默认参数: 精度: 83.33
"""
nb = GaussianNB()
nb.fit(X_train, y_train)
y_predict = nb.predict(X_test)
true_num = (y_predict.astype(y_test.dtype) == y_test).sum()
nb_accuracy = true_num / len(y_test)
print(f'贝叶斯精度: {nb_accuracy * 100} %')

# 7. SVM
"""
默认参数kernel=rbf, degree=3, gamma='scale': 精度: 99.11%
kernel='linear': 精度: 97.11%
kernel='poly': 精度: 98.89%
kernel='sigmoid': 精度: 90.89%

kernel='rbf', degree=1: 精度: 99.11%
kernel='rbf', degree=5: 精度: 99.11%

kernel='rbf', degree=3, gamma='auto': 精度: 48.67%
"""
svmc = SVC()
svmc.fit(X_train, y_train)
y_predict = svmc.predict(X_test)
true_num = (y_predict.astype(y_test.dtype) == y_test).sum()
svm_accuracy = true_num / len(y_test)
print(f'SVM精度: {svm_accuracy * 100} %')

# 整合
algorithms = [KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, SVC]
algorithm_names = []
accuracy_results = []
for algorithm in algorithms:
    algorithm_name = algorithm.mro()[0].__name__
    algorithm_names.append(algorithm_name)
    clf = algorithm()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    accuracy_results.append(accuracy)

fig = plt.figure(figsize=(15, 8), dpi=80)
vbar = plt.bar(range(len(algorithms)), accuracy_results)
plt.xticks(range(len(algorithms)), algorithm_names, rotation=0)
plt.bar_label(vbar, labels=['%.2f %s'%(value * 100, '%') for value in accuracy_results], padding=5, color='b', fontsize=14) # 添加注解
plt.ylabel('accuracy')
plt.title('The comparison between five classification algorithms')
plt.show()
