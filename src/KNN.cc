#include "KNN.h"

// PRIVATE

constexpr float KNN::_square(float x) { return x * x; }

inline float KNN::_euclidean_distance(
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
        [this](float a, float b){ float diff = a - b; return _square(diff); }
    );

    return calculate_root ? std::sqrt(distance) : distance;

}

float KNN::_get_majority(const std::vector<float>& query) {
    using Pair = std::pair<float, float>;  // (distance, class)

    // Thread-local heaps to avoid contention
    std::vector<std::priority_queue<Pair>> local_heaps(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& heap = local_heaps[tid];

        #pragma omp for nowait
        for (size_t i = 0; i < _num_samples; ++i) {
            float d = _euclidean_distance(query, _X[i], false);

            if (heap.size() < _k) {
                heap.emplace(d, _y[i]);  
            } else if (d < heap.top().first) {
                heap.pop();  // Remove farthest
                heap.emplace(d, _y[i]);
            }
        }
    }

    // Merge all heaps into one global min-heap
    std::priority_queue<Pair> global_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            if (global_heap.size() < _k) {
                global_heap.push(heap.top());
            } else if (heap.top().first < global_heap.top().first) {
                global_heap.pop();
                global_heap.push(heap.top());
            }
            heap.pop();
        }
    }

    // Voting among k nearest neighbors
    std::unordered_map<float, size_t> votes;
    while (!global_heap.empty()) {
        votes[global_heap.top().second]++;
        global_heap.pop();
    }

    return std::max_element(votes.begin(), votes.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}



// PUBLIC
KNN::KNN(
    const std::vector<std::vector<float>>& X, 
    const std::vector<float>& y, 
    const size_t& K
)
    : _X(X), 
      _y(y) {

    if(K == 0) { _k = std::sqrt(X.size()); }
    else { _k = K; }

    _num_samples = X.size();
    _num_features = X[0].size();

    std::unordered_set<float> unique_set(y.begin(), y.end());
    _classes = std::vector<float>(unique_set.begin(), unique_set.end());
}


KNN::KNN(const KNN& other) 
    : _X(other._X), 
      _y(other._y), 
      _k(other._k), 
      _num_features(other._num_features), 
      _num_samples(other._num_samples), 
      _classes(other._classes){};


float KNN::operator()(const std::vector<float>& X) {

    return _get_majority(X);

}


std::vector<float> KNN::operator()(const std::vector<std::vector<float>>& X_test) {

    std::vector<float> predictions(X_test.size());
    
    std::transform(
        X_test.begin(), 
        X_test.end(), 
        predictions.begin(), 
        [this](const std::vector<float>& sample) { return (*this)(sample); }
    );    

    return predictions;
}