#include "KNN.h"

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

float KNN::_get_majority(const std::vector<std::pair<float, float>>& distances){
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


// PUBLIC
KNN::KNN(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const size_t& K): _X(X), _y(y) {
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
      _num_samples(other._num_samples), _classes(other._classes){};


float KNN::operator()(const std::vector<float>& X) {
    std::vector<std::pair<float, float>> distances(_num_samples);

    // Compute distances
    for (size_t i = 0; i < _num_samples; i++) {
        distances[i] = {_euclidean_distance(X, _X[i], false), _y[i]};
    }

    // Sort by distance (ascending)
    std::sort(
        distances.begin(), 
        distances.end(),
        [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            return a.first < b.first;
        }
    );

    return _get_majority(distances);
}


std::vector<float> KNN::operator()(const std::vector<std::vector<float>>& X) {
    std::vector<float> predictions(X.size());  

    for(size_t i = 0; i < X.size(); i++){
        predictions[i] = (*this)(X[i]);
    }

    return predictions;
}