cmake_minimum_required(VERSION 3.10)
project(KNNProject)

# Set compiler to clang++
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -Ofast -ftree-vectorize -march=native -fopenmp")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Create executable
add_executable(knn_app
    src/main.cc
    src/KNN.cc
    src/CentroidClassifier.cc
)

# Include directories
target_include_directories(knn_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Add compile options
target_compile_options(knn_app PRIVATE
    -Wall
    -Wextra
)