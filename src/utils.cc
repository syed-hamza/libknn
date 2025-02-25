#include <iostream>
#include <vector>

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
    const int num_iterations
){

    std::vector<double> train_times, predict_times;

    for (int i = 0; i < num_iterations; i++){
        // Start timing training
        auto start_train = std::chrono::high_resolution_clock::now();
        KNN model(X_train, y_train, 0); // Training
        auto end_train = std::chrono::high_resolution_clock::now();

        // Calculate training time
        double train_time = std::chrono::duration<double, std::milli>(end_train - start_train).count();
        train_times.push_back(train_time);

        // Start timing prediction
        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<float> preds = model(X_test); // Prediction
        auto end_pred = std::chrono::high_resolution_clock::now();

        // Calculate prediction time
        double pred_time = std::chrono::duration<double, std::milli>(end_pred - start_pred).count();
        predict_times.push_back(pred_time);
    }

    // Compute mean and standard deviation
    auto mean_std = [](const std::vector<double> &times){
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = std::accumulate(
            times.begin(), 
            times.end(), 
            0.0,
            [mean](double acc, double x){ return acc + (x - mean) * (x - mean); }
        ); 
        variance /= times.size();
        double std_dev = std::sqrt(variance);
        return std::make_pair(mean, std_dev);
    };

    auto [train_mean, train_std] = mean_std(train_times);
    auto [pred_mean, pred_std] = mean_std(predict_times);

    // Print formatted results
    std::cout << "===== Experiment Parameters =====" << std::endl;
    std::cout << std::left << std::setw(24) << "Number of samples" << ": " << X_train.size() + X_test.size() << std::endl;
    std::cout << std::left << std::setw(24) << "Number of features" << ": " << X_train[0].size() << std::endl;
    std::cout << std::left << std::setw(24) << "Number of iterations" << ": " << num_iterations << std::endl;
    std::cout << std::left << std::setw(24) << "Algorithm" << ": brute" << std::endl;

    std::cout << "===== Performance Statistics =====" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(24) << "Fit Time (Mean)" << ": " << std::setw(8) << train_mean << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Fit Time (Std Dev)" << ": " << std::setw(8) << train_std << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Mean)" << ": " << std::setw(8) << pred_mean << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Std Dev)" << ": " << std::setw(8) << pred_std << " ms" << std::endl;
}