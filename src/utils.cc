#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <chrono>

std::pair<double, std::string> choose_time_unit(double time_in_ms) {
    if (time_in_ms < 0.001) {
        // Convert to nanoseconds
        return {time_in_ms * 1'000'000.0, "ns"};
    } else if (time_in_ms < 1.0) {
        // Convert to microseconds
        return {time_in_ms * 1'000.0, "µs"};
    } else {
        // Keep as milliseconds
        return {time_in_ms, "ms"};
    }
}

std::vector<std::vector<float>> reshape(
    const std::vector<float> &flat_matrix,
    const size_t &num_columns)
{
    if (flat_matrix.size() % num_columns != 0)
    {
        throw std::invalid_argument("The flat matrix cannot be divided evenly");
    }

    size_t num_rows = flat_matrix.size() / num_columns;
    std::vector<std::vector<float>> reshaped_matrix(num_rows, std::vector<float>(num_columns));

    for (size_t i = 0; i < num_rows; i++)
    {
        for (size_t j = 0; j < num_columns; j++)
        {
            reshaped_matrix[i][j] = flat_matrix[i * num_columns + j];
        }
    }

    return reshaped_matrix;
}

void bench(
    const std::vector<std::vector<float>> &X_train,
    const std::vector<float> &y_train,
    const std::vector<std::vector<float>> &X_test,
    const int num_iterations)
{
    std::vector<double> train_times, predict_times;

    for (int i = 0; i < num_iterations; i++){
        auto start_train = std::chrono::high_resolution_clock::now();
        // CentroidClassifier model(X_train, y_train);
        KNN model(X_train, y_train, 5);
        auto end_train = std::chrono::high_resolution_clock::now();
        double train_time = std::chrono::duration<double, std::milli>(end_train - start_train).count();
        train_times.push_back(train_time);

        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<float> preds = model(X_test);
        auto end_pred = std::chrono::high_resolution_clock::now();
        double pred_time = std::chrono::duration<double, std::milli>(end_pred - start_pred).count();
        predict_times.push_back(pred_time);
    }

    auto mean_std = [](const std::vector<double> &times){
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = std::accumulate(
            times.begin(), times.end(), 0.0,
            [mean](double acc, double x){ return acc + (x - mean) * (x - mean); }
        ) / times.size();
        return std::make_pair(mean, std::sqrt(variance));
    };

    auto [train_mean, train_std] = mean_std(train_times);
    auto [pred_mean, pred_std] = mean_std(predict_times);

    double train_time_per_sample = train_mean / X_train.size();
    double predict_time_per_sample = pred_mean / X_test.size();

    // Dynamically choose time units
    auto [adjusted_train_mean, train_unit] = choose_time_unit(train_mean);
    auto [adjusted_train_std, _] = choose_time_unit(train_std);
    auto [adjusted_pred_mean, pred_unit] = choose_time_unit(pred_mean);
    auto [adjusted_pred_std, __] = choose_time_unit(pred_std);
    auto [adjusted_train_time_per_sample, train_per_sample_unit] = choose_time_unit(train_time_per_sample);
    auto [adjusted_predict_time_per_sample, pred_per_sample_unit] = choose_time_unit(predict_time_per_sample);

    std::cout << "===== Experiment Parameters =====" << std::endl;
    std::cout << std::left << std::setw(30) << "Number of Training Samples" << ": " << X_train.size() << std::endl;
    std::cout << std::left << std::setw(30) << "Number of Testing Samples" << ": " << X_test.size() << std::endl;
    std::cout << std::left << std::setw(30) << "Total Number of Samples" << ": " << X_train.size() + X_test.size() << std::endl;
    std::cout << std::left << std::setw(30) << "Number of features" << ": " << X_train[0].size() << std::endl;
    std::cout << std::left << std::setw(30) << "Number of iterations" << ": " << num_iterations << std::endl;
    std::cout << std::left << std::setw(30) << "Algorithm" << ": brute" << std::endl;

    std::cout << "===== Performance Statistics =====" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(24) << "Fit Time (Mean)" 
              << ": " << std::setw(8) << adjusted_train_mean << " " << train_unit << std::endl;
    std::cout << std::left << std::setw(24) << "Fit Time (Std Dev)" 
              << ": " << std::setw(8) << adjusted_train_std << " " << train_unit << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Mean)" 
              << ": " << std::setw(8) << adjusted_pred_mean << " " << pred_unit << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Std Dev)" 
              << ": " << std::setw(8) << adjusted_pred_std << " " << pred_unit << std::endl;
    std::cout << std::left << std::setw(24) << "Train Time per Sample" 
              << ": " << std::setw(8) << adjusted_train_time_per_sample << " " << train_per_sample_unit << "/sample" << std::endl;
    std::cout << std::left << std::setw(24) << "Predict Time per Sample" 
              << ": " << std::setw(8) << adjusted_predict_time_per_sample << " " << pred_per_sample_unit << "/sample" << std::endl;
}