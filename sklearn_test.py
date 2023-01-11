import sklearn
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

# dataset = load_boston(return_X_y=True)
# features, target = load_boston(return_X_y=True)

# iris_dataset = load_iris()

# iris_features, iris_class = load_iris(return_X_y=True)

# digits_dataset = load_digits()
# print(digits_dataset.data.shape) # (1797, 64)

# digits_features, digits_class = load_digits(return_X_y=True, n_class=6)
# print(digits_features.data.shape) # (1083, 64)
# print(digits_class.data.shape) # (1083,)


dataset = load_breast_cancer()
print(dataset.data.shape) # (442, 10)
print(dataset.feature_names)

features, n_class = load_breast_cancer(return_X_y=True)
print(features.data.shape) # (442, 10)
print(n_class.data.shape) # (1442,)
