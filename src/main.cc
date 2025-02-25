#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <unordered_set>
#include <npy.hpp>

#include "KNN.h"
#include "utils.cc"
#include "metrics.cc"



int main()
{

    const size_t num_features = 10;
    const size_t num_samples = 1000;
    const int num_iterations = 500; // Number of iterations for benchmarking

    std::string X_path = "../data/X.npy";
    std::string y_path = "../data/y.npy";

    auto X_flat = npy::read_npy<float>(X_path);
    auto y_flat = npy::read_npy<float>(y_path);

    std::vector<std::vector<float>> X = reshape(X_flat.data, num_features);
    std::vector<float> y = y_flat.data;

    const size_t num_train = 800;

    std::vector<std::vector<float>> X_train;
    std::vector<float> y_train;
    std::vector<std::vector<float>> X_test;
    std::vector<float> y_test;

    for (size_t i = 0; i < num_samples; i++)
    {
        if (i < num_train)
        {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        }
        else
        {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
    }

    // Call benchmark function
    bench(X_train, y_train, X_test, num_iterations);

    // Train and predict
    KNN model(X_train, y_train, 0);
    std::vector<float> preds = model(X_test);

    classification_report(preds, y_test);

    // // Print results with color formatting
    // std::cout << "\033[1mTruth  : Predicted\033[0m\n"; // Bold header
    // for(size_t i = 0; i < y_test.size(); i++){
    //     if (static_cast<int>(y_test[i]) == static_cast<int>(preds[i])) {
    //         std::cout << "\033[32m"; // Green for correct predictions
    //     } else {
    //         std::cout << "\033[31m"; // Red for incorrect predictions
    //     }
    //     std::cout << static_cast<int>(y_test[i]) << " : " << static_cast<int>(preds[i]) << "\033[0m" << std::endl;
    // }

    return 0;
}
