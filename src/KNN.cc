#include "KNN.h"

// PRIVATE
float KNN::_sum(const std::vector<float>& vec) {
    float sum = 0.0f;
    
    for(const float x : vec)
        sum += x;
    
    return sum;
}

float KNN::_dot(const std::vector<float>& A, const std::vector<float>& B) {
    if(A.size() != B.size()) {
        throw std::invalid_argument("Lenghts of vectors not equal");
    }
    
    float sum = 0.0f;
    
    for(size_t i = 0; i < A.size(); i++)
        sum += (A[i] * B[i]);

    return sum;
}

std::vector<float> KNN::_square(const std::vector<float>& vec){
    std::vector<float> res(vec.size());

    for(size_t i = 0; i < vec.size(); i++)
        res[i] = vec[i] * vec[i];

    return res;
}

// PUBLIC
KNN::KNN() : _num_features(0), _num_samples(0) {
    _feature_target_pairs.reserve(_num_samples);
}

KNN::KNN(const std::uint32_t k) : _k(k), _num_features(0), _num_samples(0) {
    _feature_target_pairs.reserve(_num_samples);
};

void KNN::fit(const std::vector<std::vector<float>>& X, const std::vector<int>& y){

}

bool KNN::predict(const std::vector<float>& X, const float y){

}

std::vector<bool> KNN::predict(const std::vector<std::vector<float>>& X, const std::vector<float>& y){
    
}