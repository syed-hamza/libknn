#ifndef CENTROIDCLASSIFIER_H
#define CENTROIDCLASSIFIER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <stdexcept>

class CentroidClassifier{
private:
    std::unordered_map<float, std::vector<float>> _class_centroids;

    void _accumulate_vector(std::vector<float>& A, const std::vector<float>& B);
    std::vector<float> _get_centroid(const std::vector<std::vector<float>>& data_points);

    float _euclidean_distance(    
        const std::vector<float>& A, 
        const std::vector<float>& B
    );
    float _manhattan_distance(
        const std::vector<float>& A, 
        const std::vector<float>& B
    );

public:
    CentroidClassifier(
        const std::vector<std::vector<float>>& X,
        const std::vector<float>& y
    );

    CentroidClassifier(const CentroidClassifier& other);

    float operator()(const std::vector<float>& X);
    std::vector<float> operator()(const std::vector<std::vector<float>>& X);
};

#endif