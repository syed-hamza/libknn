#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <nvml.h>  // NVIDIA Management Library
#include <chrono>  // For timing operations
#include "KNN_cuda.h"

// Structure to hold GPU metrics
struct GPUMetrics {
    unsigned int gpu_utilization;
    unsigned int memory_utilization;
    size_t memory_used;
    size_t memory_total;
    double data_transfer_time_ms;
};

// Global tracking variables for batched operations
static bool g_first_operation = true;
static size_t g_total_operations = 0;
static double g_total_transfer_time = 0.0;
static double g_total_kernel_time = 0.0;
static double g_total_result_transfer_time = 0.0;
static size_t g_total_data_transferred_kb = 0;
static unsigned int g_peak_gpu_utilization = 0;
static unsigned int g_peak_memory_utilization = 0;

// Flag to track if we've already shown GPU metrics in a benchmark run
static bool g_metrics_already_shown = false;

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

// CUDA kernel for computing Euclidean distances
__global__ void computeDistancesKernel(
    const float* query_point,
    const float* train_points,
    float* distances,
    int num_train_samples,
    int num_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_train_samples) {
        float distance = 0.0f;
        
        for (int j = 0; j < num_features; j++) {
            float diff = query_point[j] - train_points[idx * num_features + j];
            distance += diff * diff;
        }
        
        distances[idx] = distance;
    }
}

// Host function to allocate memory and launch kernel
void computeDistancesGPU(
    const std::vector<float>& query_point,
    const std::vector<std::vector<float>>& train_points,
    std::vector<float>& distances)
{
    // Initialize NVML if this is the first operation
    static bool nvmlInitialized = false;
    static GPUMetrics initialMetrics = {};
    
    if (!nvmlInitialized) {
        nvmlInitialized = initNVML();
        if (nvmlInitialized) {
            initialMetrics = getGPUMetrics();
            
            // Only print initial metrics on the first operation and if not already shown in a benchmark
            if (g_first_operation && !g_metrics_already_shown) {
                std::cout << "\n===== Initial GPU State =====" << std::endl;
                std::cout << "GPU Utilization: " << initialMetrics.gpu_utilization << "%" << std::endl;
                std::cout << "Memory Utilization: " << initialMetrics.memory_utilization << "%" << std::endl;
                std::cout << "Memory Used/Total: " << initialMetrics.memory_used / (1024 * 1024) 
                          << " / " << initialMetrics.memory_total / (1024 * 1024) << " MB" << std::endl;
            }
        }
    }
    
    // Only check metrics on first operation and if not already shown in a benchmark
    bool shouldReportMetrics = g_first_operation && !g_metrics_already_shown;
    GPUMetrics beforeMetrics = {};
    
    if (shouldReportMetrics && nvmlInitialized) {
        beforeMetrics = getGPUMetrics();
    }
    
    int num_train_samples = train_points.size();
    int num_features = query_point.size();
    
    // Start timing for data transfer
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    // Allocate and copy query point to device
    float* d_query_point;
    cudaMalloc(&d_query_point, num_features * sizeof(float));
    cudaMemcpy(d_query_point, query_point.data(), num_features * sizeof(float), cudaMemcpyHostToDevice);
    
    // Flatten and copy training points to device
    std::vector<float> flattened_train_points;
    flattened_train_points.reserve(num_train_samples * num_features);
    
    for (const auto& sample : train_points) {
        flattened_train_points.insert(flattened_train_points.end(), sample.begin(), sample.end());
    }
    
    float* d_train_points;
    cudaMalloc(&d_train_points, flattened_train_points.size() * sizeof(float));
    cudaMemcpy(d_train_points, flattened_train_points.data(), 
               flattened_train_points.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for distances on device
    float* d_distances;
    cudaMalloc(&d_distances, num_train_samples * sizeof(float));
    
    // End timing for data transfer
    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    
    // Track total data transferred
    g_total_transfer_time += transfer_time;
    size_t data_size_kb = (num_features + num_train_samples * num_features) * sizeof(float) / 1024.0;
    g_total_data_transferred_kb += data_size_kb;
    
    // Only get GPU metrics after transfer for the first operation and if not already shown in a benchmark
    if (shouldReportMetrics && nvmlInitialized) {
        GPUMetrics afterTransferMetrics = getGPUMetrics();
        
        // Report first data transfer metrics
        std::cout << "\n===== First Data Transfer =====" << std::endl;
        std::cout << "Data Transfer Time: " << transfer_time << " ms" << std::endl;
        std::cout << "Data Size: " << data_size_kb << " KB" << std::endl;
        std::cout << "GPU Utilization: " << afterTransferMetrics.gpu_utilization << "%" << std::endl;
        std::cout << "Memory Utilization: " << afterTransferMetrics.memory_utilization << "%" << std::endl;
        
        // Update peak utilization
        g_peak_gpu_utilization = std::max(g_peak_gpu_utilization, afterTransferMetrics.gpu_utilization);
        g_peak_memory_utilization = std::max(g_peak_memory_utilization, afterTransferMetrics.memory_utilization);
    }
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_train_samples + blockSize - 1) / blockSize;
    
    // Start kernel timing
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    computeDistancesKernel<<<numBlocks, blockSize>>>(
        d_query_point, d_train_points, d_distances, num_train_samples, num_features);
    
    // Ensure kernel is complete
    cudaDeviceSynchronize();
    
    // End kernel timing
    auto kernel_end = std::chrono::high_resolution_clock::now();
    double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    g_total_kernel_time += kernel_time;
    
    // Get GPU metrics during kernel execution only for first operation
    if (shouldReportMetrics && nvmlInitialized) {
        GPUMetrics duringKernelMetrics = getGPUMetrics();
        
        // Update peak utilization
        g_peak_gpu_utilization = std::max(g_peak_gpu_utilization, duringKernelMetrics.gpu_utilization);
        g_peak_memory_utilization = std::max(g_peak_memory_utilization, duringKernelMetrics.memory_utilization);
    }
    
    // Start timing for result transfer
    auto result_transfer_start = std::chrono::high_resolution_clock::now();
    
    // Copy results back to host
    distances.resize(num_train_samples);
    cudaMemcpy(distances.data(), d_distances, num_train_samples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // End timing for result transfer
    auto result_transfer_end = std::chrono::high_resolution_clock::now();
    double result_transfer_time = std::chrono::duration<double, std::milli>(
        result_transfer_end - result_transfer_start).count();
    g_total_result_transfer_time += result_transfer_time;
    
    // Free device memory
    cudaFree(d_query_point);
    cudaFree(d_train_points);
    cudaFree(d_distances);
    
    // Update operation tracking
    g_first_operation = false;
    g_total_operations++;
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
    }
}

// Batch version for processing multiple query points
void computeDistancesBatchGPU(
    const std::vector<std::vector<float>>& query_points,
    const std::vector<std::vector<float>>& train_points,
    std::vector<std::vector<float>>& all_distances)
{
    // Only print metrics if they haven't been shown already in a benchmark run
    bool shouldShowMetrics = !g_metrics_already_shown;
    
    // Initialize NVML for batch processing
    bool nvmlInitialized = initNVML();
    
    if (nvmlInitialized && shouldShowMetrics) {
        GPUMetrics initialMetrics = getGPUMetrics();
        std::cout << "\n===== GPU Batch Processing Starting =====" << std::endl;
        std::cout << "GPU Utilization: " << initialMetrics.gpu_utilization << "%" << std::endl;
        std::cout << "Memory Used/Total: " << initialMetrics.memory_used / (1024 * 1024) 
                  << " / " << initialMetrics.memory_total / (1024 * 1024) << " MB" << std::endl;
    }
    
    all_distances.resize(query_points.size());
    
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    // Reset global tracking variables for this batch
    g_first_operation = true;
    g_total_operations = 0;
    g_total_transfer_time = 0.0;
    g_total_kernel_time = 0.0;
    g_total_result_transfer_time = 0.0;
    g_total_data_transferred_kb = 0;
    g_peak_gpu_utilization = 0;
    g_peak_memory_utilization = 0;
    
    // Use a larger checkpoint interval to reduce output
    const size_t checkpoint_interval = 1000; 
    
    for (size_t i = 0; i < query_points.size(); i++) {
        computeDistancesGPU(query_points[i], train_points, all_distances[i]);
        
        // Print progress indicators at larger intervals, but only if showing metrics
        if (shouldShowMetrics && i > 0 && i % checkpoint_interval == 0) {
            std::cout << "Processing: " << i << "/" << query_points.size() 
                      << " items (" << (i * 100 / query_points.size()) << "%)" << std::endl;
            
            // Only print a mid-point GPU summary for very large batches
            if (query_points.size() > 5000 && i >= query_points.size() / 2) {
                GPUMetrics midMetrics = getGPUMetrics();
                std::cout << "  - Mid-point GPU utilization: " << midMetrics.gpu_utilization 
                          << "%, Memory: " << midMetrics.memory_utilization << "%" << std::endl;
            }
        }
    }
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    
    if (nvmlInitialized && shouldShowMetrics) {
        GPUMetrics finalMetrics = getGPUMetrics();
        std::cout << "\n===== GPU Batch Processing Summary =====" << std::endl;
        std::cout << "Peak GPU Utilization: " << g_peak_gpu_utilization << "%" << std::endl;
        std::cout << "Peak Memory Utilization: " << g_peak_memory_utilization << "%" << std::endl;
        std::cout << "Total Data Transferred: " << g_total_data_transferred_kb / 1024.0 << " MB" << std::endl;
        std::cout << "Total Processing Time: " << batch_time << " ms" << std::endl;
        std::cout << "Avg Time Per Item: " << batch_time / query_points.size() << " ms" << std::endl;
        std::cout << "Avg Data Transfer: " << g_total_transfer_time / g_total_operations << " ms" << std::endl;
        std::cout << "Avg Kernel Execution: " << g_total_kernel_time / g_total_operations << " ms" << std::endl;
        std::cout << "Avg Result Transfer: " << g_total_result_transfer_time / g_total_operations << " ms" << std::endl;
        
        // Mark metrics as already shown for future benchmark runs
        g_metrics_already_shown = true;
        
        // Shutdown NVML
        shutdownNVML();
    }
}

// Reset GPU metrics tracking status (for benchmarking)
void resetGPUMetricsTracking() {
    // Reset the flag so metrics will be shown again next time
    g_metrics_already_shown = false;
    
    // Reset counters
    g_first_operation = true;
    g_total_operations = 0;
    g_total_transfer_time = 0.0;
    g_total_kernel_time = 0.0;
    g_total_result_transfer_time = 0.0;
    g_total_data_transferred_kb = 0;
    g_peak_gpu_utilization = 0;
    g_peak_memory_utilization = 0;
} 