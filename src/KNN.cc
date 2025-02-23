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

float KNN::_get_majority(std::vector<std::pair<float, float>>& distances) {
    // Use nth_element to get the top-k smallest distances
    std::nth_element(distances.begin(), distances.begin() + _k, distances.end(),
                     [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                         return a.first < b.first;
                     });

    // Count occurrences of each class among the top-k
    std::unordered_map<float, size_t> votes;
    float predicted_class = -1;
    size_t max_votes = 0;

    for (size_t i = 0; i < _k; i++) {
        float label = distances[i].second;
        votes[label]++;

        if (votes[label] > max_votes) {
            max_votes = votes[label];
            predicted_class = label;
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


std::vector<float> KNN::operator()(const std::vector<std::vector<float>>& X_test) {
    std::vector<float> predictions(X_test.size());

    for (size_t test_idx = 0; test_idx < X_test.size(); test_idx++) {
        std::vector<std::pair<float, float>> distances(_num_samples);

        // Compute distances for this test point
        for (size_t i = 0; i < _num_samples; i++) {
            distances[i] = {_euclidean_distance(X_test[test_idx], _X[i], false), _y[i]};
        }

        // Sort and get majority
        std::nth_element(distances.begin(), distances.begin() + _k, distances.end(),
                         [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                             return a.first < b.first;
                         });

        predictions[test_idx] = _get_majority(distances);
    }

    return predictions;
}
