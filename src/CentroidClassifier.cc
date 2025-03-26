#include "CentroidClassifier.h"

// Private
void CentroidClassifier::_accumulate_vector(std::vector<float>& A, const std::vector<float>& B){
    for(size_t i = 0; i < A.size(); i++){
        A[i] += B[i];
    }
}

std::vector<float> CentroidClassifier::_get_centroid(const std::vector<std::vector<float>>& data_points){
    size_t num_features = data_points[0].size();
    size_t num_samples = data_points.size();

    std::vector<float> centroid(num_features, 0.0f);

    for(size_t i = 0; i < num_samples; i++){ _accumulate_vector(centroid, data_points[i]); }

    for(size_t i = 0; i < num_features; i++){ centroid[i] /= num_samples; }

    return centroid;
}

float CentroidClassifier::_euclidean_distance(
    const std::vector<float>& A, 
    const std::vector<float>& B
) {

    float distance = 0.0f;

    for (size_t i = 0; i < A.size(); i++) {
        float diff = A[i] - B[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

float CentroidClassifier::_manhattan_distance(
    const std::vector<float>& A, 
    const std::vector<float>& B
) {
   
    float distance = 0.0f;
    for (size_t i = 0; i < A.size(); i++) {
        distance += std::abs(A[i] - B[i]);
    }

    return distance;
}


// Public
CentroidClassifier::CentroidClassifier(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y
) {
    if(y.size() != X.size()){
        throw std::invalid_argument("Number of samples not equal to number of classes");
    }

    std::unordered_map<float, std::vector<std::vector<float>>> class_samples_groups;

    for(size_t i = 0; i < y.size(); i++){
        class_samples_groups[y[i]].push_back(X[i]);
    }

    for(const auto& pair : class_samples_groups){
        _class_centroids[pair.first] = _get_centroid(pair.second);
    }
}

CentroidClassifier::CentroidClassifier(const CentroidClassifier& other) : _class_centroids(other._class_centroids) {};

float CentroidClassifier::operator()(const std::vector<float>& X){
    float best_class = -1;
    float min_distance = std::numeric_limits<float>::max();

    for (const auto& pair : _class_centroids) {
        float distance = _euclidean_distance(X, pair.second);
        if (distance < min_distance) {
            min_distance = distance;
            best_class = pair.first;
        }
    }

    return best_class;
}


std::vector<float> CentroidClassifier::operator()(const std::vector<std::vector<float>>& X){
    std::vector<float> predictions(X.size());

    std::transform(
        X.begin(), 
        X.end(), 
        predictions.begin(), 
        [this](const std::vector<float>& sample) { return (*this)(sample); }
    );    

    return predictions;
}