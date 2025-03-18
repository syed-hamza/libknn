#ifndef KNN_CUDA_H
#define KNN_CUDA_H

#include <vector>

// Compute distances between a single query point and all training points
void computeDistancesGPU(
    const std::vector<float>& query_point,
    const std::vector<std::vector<float>>& train_points,
    std::vector<float>& distances);

// Compute distances for multiple query points (batch processing)
void computeDistancesBatchGPU(
    const std::vector<std::vector<float>>& query_points,
    const std::vector<std::vector<float>>& train_points,
    std::vector<std::vector<float>>& all_distances);

// Reset GPU metrics tracking status (for benchmarking)
void resetGPUMetricsTracking();

#endif // KNN_CUDA_H 