#include <iostream>
#include <npy.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <map>

class Sample{
public:
    float square_norm;
    int target;

    Sample() : square_norm(0.0f), target(-1.0f) {};

    Sample(const float square_norm, const int target)
        : square_norm(square_norm), target(target) {};

    bool operator<(const Sample& other) const {
        return square_norm < other.square_norm;
    }
};

void parse_npy(
    const std::vector<float>& flat_matrix, 
    std::vector<std::vector<float>>& matrix, 
    const size_t rows,
    const size_t cols
){
    if (rows * cols != flat_matrix.size()) {
        throw std::invalid_argument("Size mismatch: Cannot reshape 1D array into given dimensions.");
    }

    for (size_t i = 0; i < rows; i++) {  
        for (size_t j = 0; j < cols; j++) {  
            matrix[i][j] = flat_matrix[i * cols + j];  
        }
    }
}

class KNN {
private:
    std::vector<Sample> _feature_target_pairs;
    size_t _num_samples;
    size_t _num_features;

    float _sum(const std::vector<float>& vec){
        float sum = 0.0f;
        for(const float x : vec)
            sum += x * x;
        return sum;
    }

    float _dot(const std::vector<float>& A, const std::vector<float>& B){
        if(A.size() != B.size()){
            throw std::invalid_argument("Lengths of vectors unequal.");
        }
        
        float sum = 0.0f;
        size_t len = A.size();
        for(size_t i = 0; i < len; i++){
            sum += A[i] * B[i];
        }
        return sum;
    }

public:
    size_t k;

    KNN(const size_t k) : k(k) {};

    void fit(
        const std::vector<std::vector<float>>& X,
        const std::vector<int>& y
    ) {
        _num_samples = y.size();
        _num_features = X[0].size();

        _feature_target_pairs.resize(_num_samples);

        for(size_t i = 0; i < _num_samples; i++){
            float square_norm = _sum(X[i]);
            _feature_target_pairs[i] = Sample(square_norm, y[i]);
        }
    }

    int predict(const std::vector<float>& X){
        std::vector<Sample> distances(_num_samples);
        for(size_t i = 0; i < _num_samples; i++){   
            distances[i] = Sample(
                _feature_target_pairs[i].square_norm - 2 * _dot(X, X) + _sum(X),
                _feature_target_pairs[i].target
            );
        }

        std::map<int, size_t> votes;
        std::sort(distances.begin(), distances.end());
        for(size_t i = 0; i < k; i++){
            votes[distances[i].target]++;
        }

        return std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};

int main(){
    std::string X_path = "data/X.npy";
    std::string y_path = "data/y.npy";

    auto X = npy::read_npy<float>(X_path);
    auto y = npy::read_npy<int>(y_path);

    std::vector<float> X_data = X.data;
    std::vector<int> y_data = y.data;

    std::vector<std::vector<float>> flat_X(10000, std::vector<float>(3));
    parse_npy(X_data, flat_X, 10000, 3);

    KNN knn(5);
    knn.fit(flat_X, y_data);

    std::vector<float> sample = {1.2, 0.5, -0.3};
    int prediction = knn.predict(sample);

    std::cout << "Predicted class: " << prediction << std::endl;
    return 0;
}