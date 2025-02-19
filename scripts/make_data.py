import numpy as np
from pprint import pprint
from sklearn.datasets import make_classification

n_samples = int(1e4)
n_features = 3
n_informative = 1
n_redundant = 1
n_classes = 2
n_clusters_pre_class = 1

dataset = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_classes=n_classes,
    n_clusters_per_class=n_clusters_pre_class
)

X = np.array(dataset[0], dtype=np.float32)
y = np.array(dataset[1], dtype=np.float32)

print(X.shape)
print(y.shape)

np.save("data/X.npy", X)
np.save("data/y.npy", y)