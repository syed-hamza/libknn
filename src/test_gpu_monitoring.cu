#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <nvml.h>  // NVIDIA Management Library
#include <chrono>  // For timing operations

// Structure to hold GPU metrics
struct GPUMetrics {
    unsigned int gpu_utilization;
    unsigned int memory_utilization;
    size_t memory_used;
    size_t memory_total;
    double data_transfer_time_ms;
};

// Initialize NVML library
bool initNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    return true;
}

// Shutdown NVML library
void shutdownNVML() {
    nvmlShutdown();
}

// Get current GPU metrics
GPUMetrics getGPUMetrics() {
    GPUMetrics metrics = {};
    nvmlDevice_t device;
    nvmlReturn_t result;
    
    // Get first device handle
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        return metrics;
    }
    
    // Get utilization rates
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        metrics.gpu_utilization = utilization.gpu;
        metrics.memory_utilization = utilization.memory;
    }
    
    // Get memory info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        metrics.memory_used = memory.used;
        metrics.memory_total = memory.total;
    }
    
    return metrics;
}

// Simple CUDA kernel that performs a computation
__global__ void simpleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform some computation to load the GPU
        float x = 0;
        for (int i = 0; i < 1000; i++) {
            x = sinf(cosf(data[idx]) + x);
        }
        data[idx] = x;
    }
}

int main() {
    std::cout << "===== GPU Monitoring Test =====" << std::endl;
    
    // Initialize NVML
    bool nvmlInitialized = initNVML();
    
    if (!nvmlInitialized) {
        std::cerr << "Failed to initialize NVML. Exiting..." << std::endl;
        return 1;
    }
    
    // Get initial GPU metrics
    GPUMetrics initialMetrics = getGPUMetrics();
    std::cout << "===== Initial GPU Metrics =====" << std::endl;
    std::cout << "GPU Utilization: " << initialMetrics.gpu_utilization << "%" << std::endl;
    std::cout << "Memory Utilization: " << initialMetrics.memory_utilization << "%" << std::endl;
    std::cout << "Memory Used: " << initialMetrics.memory_used / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Memory Total: " << initialMetrics.memory_total / (1024 * 1024) << " MB" << std::endl;
    
    // Allocate memory on host and device
    const int n = 10 * 1024 * 1024;  // 10 million elements
    std::vector<float> h_data(n, 1.0f);
    float* d_data = nullptr;
    
    // Start timing for data transfer
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // End timing for data transfer
    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    
    std::cout << "\n===== After Memory Allocation and Transfer =====" << std::endl;
    std::cout << "Data Transfer Time: " << transfer_time << " ms" << std::endl;
    std::cout << "Data Size Transferred: " << (n * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Get metrics after transfer
    GPUMetrics afterTransferMetrics = getGPUMetrics();
    std::cout << "GPU Utilization: " << afterTransferMetrics.gpu_utilization << "%" << std::endl;
    std::cout << "Memory Utilization: " << afterTransferMetrics.memory_utilization << "%" << std::endl;
    std::cout << "Memory Used: " << afterTransferMetrics.memory_used / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Memory Delta: " << 
        (afterTransferMetrics.memory_used - initialMetrics.memory_used) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Get metrics before kernel execution
    std::cout << "\n===== Before Kernel Execution =====" << std::endl;
    GPUMetrics beforeKernelMetrics = getGPUMetrics();
    std::cout << "GPU Utilization: " << beforeKernelMetrics.gpu_utilization << "%" << std::endl;
    std::cout << "Memory Utilization: " << beforeKernelMetrics.memory_utilization << "%" << std::endl;
    
    // Start kernel timing
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    // Execute kernel multiple times to ensure load
    for (int i = 0; i < 5; i++) {
        simpleKernel<<<numBlocks, blockSize>>>(d_data, n);
        cudaDeviceSynchronize();
        
        // Get metrics during kernel execution (after each iteration)
        if (i == 2) {  // Check metrics in the middle of iterations
            GPUMetrics duringKernelMetrics = getGPUMetrics();
            std::cout << "\n===== During Kernel Execution (Iteration " << i << ") =====" << std::endl;
            std::cout << "GPU Utilization: " << duringKernelMetrics.gpu_utilization << "%" << std::endl;
            std::cout << "Memory Utilization: " << duringKernelMetrics.memory_utilization << "%" << std::endl;
        }
    }
    
    // End kernel timing
    auto kernel_end = std::chrono::high_resolution_clock::now();
    double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    
    // Get metrics after kernel execution
    GPUMetrics afterKernelMetrics = getGPUMetrics();
    std::cout << "\n===== After Kernel Execution =====" << std::endl;
    std::cout << "Kernel Execution Time: " << kernel_time << " ms" << std::endl;
    std::cout << "GPU Utilization: " << afterKernelMetrics.gpu_utilization << "%" << std::endl;
    std::cout << "Memory Utilization: " << afterKernelMetrics.memory_utilization << "%" << std::endl;
    
    // Start timing for result transfer
    auto result_transfer_start = std::chrono::high_resolution_clock::now();
    
    // Copy results back to host
    cudaMemcpy(h_data.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // End timing for result transfer
    auto result_transfer_end = std::chrono::high_resolution_clock::now();
    double result_transfer_time = std::chrono::duration<double, std::milli>(
        result_transfer_end - result_transfer_start).count();
    
    std::cout << "\n===== After Result Transfer =====" << std::endl;
    std::cout << "Result Transfer Time: " << result_transfer_time << " ms" << std::endl;
    std::cout << "Result Size Transferred: " << (n * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Free device memory
    cudaFree(d_data);
    
    // Get final GPU metrics
    GPUMetrics finalMetrics = getGPUMetrics();
    std::cout << "\n===== Final GPU Metrics =====" << std::endl;
    std::cout << "GPU Utilization: " << finalMetrics.gpu_utilization << "%" << std::endl;
    std::cout << "Memory Utilization: " << finalMetrics.memory_utilization << "%" << std::endl;
    std::cout << "Memory Used: " << finalMetrics.memory_used / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Operation Time: " << 
        (transfer_time + kernel_time + result_transfer_time) << " ms" << std::endl;
    
    // Print summary
    std::cout << "\n===== GPU Monitoring Summary =====" << std::endl;
    std::cout << "Host to Device Transfer: " << transfer_time << " ms, " 
              << (n * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Kernel Execution: " << kernel_time << " ms" << std::endl;
    std::cout << "Device to Host Transfer: " << result_transfer_time << " ms, " 
              << (n * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Peak GPU Utilization: " << 
        std::max(std::max(afterTransferMetrics.gpu_utilization, afterKernelMetrics.gpu_utilization), 
                 finalMetrics.gpu_utilization) << "%" << std::endl;
    
    // Shutdown NVML
    shutdownNVML();
    
    std::cout << "\nGPU monitoring test completed successfully." << std::endl;
    
    return 0;
} 