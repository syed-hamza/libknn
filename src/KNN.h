#ifndef KNN_H
#define KNN_H

#include "Sample.h"
#include <vector>
#include <iostream>
#include <cstdint>

class KNN {
private:
    std::vector<Sample> _feature_target_pairs;
    size_t _num_samples;
    size_t _num_features;

    std::uint32_t _k;

    float _sum(const std::vector<float>& vec);
    float _dot(const std::vector<float>& A, const std::vector<float>& B);
    
    std::vector<float> _square(const std::vector<float>& vec);

public:
    KNN();

    KNN(const std::uint32_t k);

    void fit(const std::vector<std::vector<float>>& X, const std::vector<int>& y);

    bool predict(const std::vector<float>& X, const float y);

    std::vector<bool> predict(const std::vector<std::vector<float>>& X, const std::vector<float>& y);

};

#endif KNN_H