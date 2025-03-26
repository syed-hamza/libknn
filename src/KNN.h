#ifndef KNN_H
#define KNN_H

#include <vector>
#include <queue>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <numeric>
#include <utility>
#include <omp.h>
#include <cstring>
#include <limits>

class KNN {
private:
    std::vector<float> _X_flat;
    std::vector<float> _y;

    size_t _k;

    size_t _num_features;
    size_t _num_samples;

    std::vector<float> _classes;

    inline float _euclidean_distance(
        const float* A, const float* B
    );

    inline float _manhattan_distance(
        const std::vector<float>& A, 
        const std::vector<float>& B
    );
    
    float _get_majority(const std::vector<float>& query);

public:

    KNN(
        const std::vector<std::vector<float>>& X, 
        const std::vector<float>& y, 
        const size_t& K=0
    );

    KNN(const KNN& other);

    float operator()(const std::vector<float>& X);
    std::vector<float> operator()(const std::vector<std::vector<float>>& X);

};

#endif