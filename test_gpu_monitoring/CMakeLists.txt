cmake_minimum_required(VERSION 3.10)
project(GPUMonitorTest LANGUAGES CXX CUDA)

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

# If NVML library not found, try to use NVIDIA's management library directly
if(NOT NVML_LIBRARY)
    set(NVML_LIBRARY "nvidia-ml")
    message(STATUS "NVML library not found in CUDA, using nvidia-ml directly")
else()
    message(STATUS "Found NVML library: ${NVML_LIBRARY}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures and standard
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_STANDARD 14)

# Create executable
add_executable(gpu_monitor_test ../src/test_gpu_monitoring.cu)

# Include directories
target_include_directories(gpu_monitor_test PRIVATE
    ${CUDA_TOOLKIT_ROOT_DIR}/include  # For NVML header
)

# Link CUDA and NVML libraries
target_link_libraries(gpu_monitor_test ${CUDA_LIBRARIES} ${NVML_LIBRARY})

# Add platform-specific compile options
if(MSVC)
    # MSVC (Visual Studio) options
    target_compile_options(gpu_monitor_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/W3">
    )
else()
    # GCC/Clang options
    target_compile_options(gpu_monitor_test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    )
endif() 