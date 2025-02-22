# Generate Data
```
mkdir data
cd data
```
Create a python file and copy paste the following code
```
import numpy as np
from sklearn.datasets import make_blobs

num_features = 100
num_samples = 1000

X, y = make_blobs(
    n_features = num_features,
    n_samples = num_samples,
    cluster_std = 7.6,
    centers = 2
)

print(f"num samples = {num_samples}")
print(f"num features = {num_features}")

X = X.astype(np.float32)
y = y.astype(np.float32)

np.save("X.npy", X)
np.save("y.npy", y)
```

# Build
```
mkdir build
cd build
cmake ..
make
```

# Execution
```
./bin/knn_app ../data/X.npy ../data/y.npy
```