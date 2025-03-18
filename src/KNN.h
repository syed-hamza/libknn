#ifndef KNN_H
#define KNN_H

#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <utility>

class KNN {
private:
    std::vector<std::vector<float>> _X;
    std::vector<float> _y;

    size_t _num_features;
    size_t _num_samples;

    std::vector<float> _classes;

    std::uint32_t _k;
    
    bool _use_gpu;

    float _euclidean_distance(const std::vector<float>& A, const std::vector<float>& B, const bool& calculate_root = true);

public:
    // Constructor with option to use GPU
    KNN(const std::vector<std::vector<float>>& X, 
        const std::vector<float>& y, 
        const size_t& K=0,
        const bool use_gpu=false);

    KNN(const KNN& other);

    float operator()(const std::vector<float>& X);
    std::vector<float> operator()(const std::vector<std::vector<float>>& X);
};

#endif