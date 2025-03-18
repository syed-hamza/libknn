#include "KNN.h"
#include "KNN_cuda.h"
#include <string>
#include <stdexcept>

// PRIVATE
float KNN::_euclidean_distance(
    const std::vector<float>& A, 
    const std::vector<float>& B, 
    const bool& calculate_root
){
    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector size mismatch: A has " + 
            std::to_string(A.size()) + ", B has " + std::to_string(B.size()));
    }

    float distance = 0.0f;

    for(size_t i = 0; i < A.size(); i++){
        float diff = A[i] - B[i];
        distance += diff * diff;
    }

    return calculate_root ? std::sqrt(distance) : distance;
}

// PUBLIC
KNN::KNN(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const size_t& K, const bool use_gpu)
    : _X(X), _y(y), _use_gpu(use_gpu) {
    if(K == 0) { _k = std::sqrt(X.size()); }
    else { _k = K; }

    _num_samples = X.size();
    _num_features = X[0].size();

    std::unordered_set<float> unique_set(y.begin(), y.end());
    _classes =  std::vector<float>(unique_set.begin(), unique_set.end());
}


KNN::KNN(const KNN& other) 
    : _X(other._X), _y(other._y), _k(other._k), 
      _num_features(other._num_features), 
      _num_samples(other._num_samples), 
      _classes(other._classes),
      _use_gpu(other._use_gpu) {};


float KNN::operator()(const std::vector<float>& X) {
    std::vector<std::pair<float, float>> distances(_num_samples);
    
    if (_use_gpu) {
        // Use GPU implementation
        std::vector<float> gpu_distances;
        computeDistancesGPU(X, _X, gpu_distances);
        
        // Map distances to pairs with corresponding labels
        for (size_t i = 0; i < _num_samples; i++) {
            distances[i] = {gpu_distances[i], _y[i]};
        }
    } else {
        // Use CPU implementation
        for (size_t i = 0; i < _num_samples; i++) {
            distances[i] = {_euclidean_distance(X, _X[i], false), _y[i]};
        }
    }

    // Sort by distance (ascending)
    std::sort(
        distances.begin(), 
        distances.end(),
        [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            return a.first < b.first;
        }
    );

    // Count votes
    std::map<float, size_t> votes;
    for (size_t i = 0; i < _k; i++) {
        votes[distances[i].second]++;
    }

    // Find the class with the most votes
    float predicted_class = -1;
    size_t max_votes = 0;

    for (const auto& [class_, count] : votes) {
        if (count > max_votes) {
            max_votes = count;
            predicted_class = class_;
        }
    }

    return predicted_class;
}


std::vector<float> KNN::operator()(const std::vector<std::vector<float>>& X) {
    std::vector<float> predictions;
    predictions.reserve(X.size());  // Reserve space for efficiency

    if (_use_gpu && X.size() > 1) {
        // Use batch GPU implementation for multiple samples
        std::vector<std::vector<float>> all_distances;
        computeDistancesBatchGPU(X, _X, all_distances);
        
        // Process each set of distances
        for (size_t sample_idx = 0; sample_idx < X.size(); sample_idx++) {
            std::vector<std::pair<float, float>> distances(_num_samples);
            
            // Map distances to pairs with corresponding labels
            for (size_t i = 0; i < _num_samples; i++) {
                distances[i] = {all_distances[sample_idx][i], _y[i]};
            }
            
            // Sort by distance (ascending)
            std::sort(
                distances.begin(), 
                distances.end(),
                [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                    return a.first < b.first;
                }
            );
            
            // Count votes
            std::map<float, size_t> votes;
            for (size_t i = 0; i < _k; i++) {
                votes[distances[i].second]++;
            }
            
            // Find the class with the most votes
            float predicted_class = -1;
            size_t max_votes = 0;
            
            for (const auto& [class_, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    predicted_class = class_;
                }
            }
            
            predictions.push_back(predicted_class);
        }
    } else {
        // Use standard implementation for each sample
        for (const auto& sample : X) {
            predictions.push_back((*this)(sample));  // Call the single-sample operator()
        }
    }

    return predictions;
}
