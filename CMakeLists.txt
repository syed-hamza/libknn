cmake_minimum_required(VERSION 3.10)
project(KNNProject LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find NVML library
find_library(NVML_LIBRARY
    NAMES nvml
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib
    PATH_SUFFIXES stubs
)

# If NVML library not found, try to use NVIDIA's library directly
if(NOT NVML_LIBRARY)
    set(NVML_LIBRARY "nvidia-ml")
    message(STATUS "NVML library not found in CUDA, using nvidia-ml directly")
else()
    message(STATUS "Found NVML library: ${NVML_LIBRARY}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_STANDARD 14)
# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -Ofast -ftree-vectorize -march=native -fopenmp")

# Define build types and optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug build
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
else()
    # Release build
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -Wall")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

# Set explicit output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")

# Create executable
add_executable(knn_app
    src/main.cc
    src/KNN.cc
    src/KNN_cuda.cu
    src/CentroidClassifier.cc
)

# Include directories
target_include_directories(knn_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include  # For NVML header
)

# Link CUDA libraries
target_link_libraries(knn_app ${CUDA_LIBRARIES} ${NVML_LIBRARY})

# Add platform-specific compile options
if(MSVC)
    # MSVC (Visual Studio) options
    target_compile_options(knn_app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/W4">
    )
else()
    # GCC/Clang options
    target_compile_options(knn_app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()