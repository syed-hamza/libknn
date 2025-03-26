# libknn: A High-Performance K-Nearest Neighbors Library  

## Overview  
**libknn** is a high-performance K-Nearest Neighbors (KNN) library optimized for modern CPUs. It utilizes **SIMD acceleration (AVX)** and **multi-core parallelism** to deliver fast and efficient computations, making it ideal for high-dimensional data in classification and regression tasks.  

## Features  
- Fast and efficient **distance calculations**  
- Optimized for **high-dimensional data**  
- Supports **multi-threading** with OpenMP  

## Installation  

### **Prerequisites**  
- A modern C++ compiler: **GCC (g++), Clang (clang++ 14+), or MSVC**  
- **CMake** (for building)  
- **OpenMP** (for multi-threading)  
- **libnpy**. The benchmark in here uses a header file to load .npy files. Refer [github.com/llohse/libnpy.git](https://github.com/llohse/libnpy.git). 

### **Installing OpenMP**  

#### **Ubuntu**  
- **For GCC:** OpenMP is included by default.  
- **For Clang:** Install OpenMP separately:  
  ```sh
  sudo apt install libomp-dev -y

### **Windows**  
- **For GCC (MinGW-w64):** Download a version with OpenMP from [winlibs.com](https://winlibs.com/).  
- **For MSVC (Visual Studio):** OpenMP is included by default, but you need to enable it:  
  1. Open **Visual Studio**.  
  2. Go to **Project Properties** → **C/C++** → **Language**.  
  3. Set **OpenMP Support** to **Enable (/openmp)**.  
  4. Click **Apply** and **OK**.  
  5. If using the command line, compile with:  
     ```sh
     cl /openmp test.cpp
     ```

---

### **macOS**  
- **Apple Clang does not support OpenMP**. You must install GCC instead:  
  ```sh
  brew install gcc
  ```
## Build

### Ubuntu
#### **Step 1: install dependencies**
#### **Step 2: Build libknn**
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```
## Benchmark
### Create mock dataset
```bash
pip install -r requirements.txt
mkdir data
cd scripts
python make_data.py
```
This script creates a dataset with clusters for classification

### Python
```bash
cd experiments
python bench_once.py
```

### libknn
```bash
cd build
./bin/knn_app
```