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

    float distance = std::inner_product(
        A.begin(), 
        A.end(), 
        B.begin(), 
        0.0f, 
        std::plus<float>(),
        [](float a, float b){ float diff = a - b; return diff * diff; }
    );

    return calculate_root ? std::sqrt(distance) : distance;

}

float KNN::_get_majority(const std::vector<float>& query) {

    using Pair = std::pair<float, float>;
    auto cmp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> max_heap(cmp);

    for (size_t i = 0; i < _num_samples; ++i) {
        float d = _euclidean_distance(query, _X[i], false);
        if (max_heap.size() < _k) {
            max_heap.push({d, _y[i]});
        } else if (d < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({d, _y[i]});
        }
    }

    // Now count votes from the k nearest neighbors
    std::unordered_map<float, size_t> votes;
    while (!max_heap.empty()) {
        votes[max_heap.top().second]++;
        max_heap.pop();
    }

    // Determine the majority vote
    float predicted_class = -1;
    size_t max_votes = 0;
    for (const auto& [label, count] : votes) {
        if (count > max_votes) {
            max_votes = count;
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

    return _get_majority(X);

}


std::vector<float> KNN::operator()(const std::vector<std::vector<float>>& X_test) {

    std::vector<float> predictions;
    predictions.reserve(X_test.size());
    
    for (const auto& sample : X_test) {
        predictions.push_back((*this)(sample));
    }
    
    return predictions;

}