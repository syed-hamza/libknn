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

## Build
```
mkdir build
cd build
cmake ..
make
```

## Execution
```
./bin/knn_app ../data/X.npy ../data/y.npy
```
# Windows:

## Build:
```
   cmake -G "Visual Studio 17 2022" -A x64 ..
   cmake --build . --config Release
``` 

## Execution(CPU):
```
 .\bin\Release\knn_app.exe ..\data\X.npy ..\data\y.npy
```
## Execution(GPU):
```
 .\bin\Release\knn_app.exe ..\data\X.npy ..\data\y.npy --gpu
```
## TODO
1) shared mem
2) Stream proccessing
3) Thrust library(sorting)
4) memory transfers