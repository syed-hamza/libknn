#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <npy.hpp>
#include <string>
#include <cstring>

#include "KNN.h"

std::vector<std::vector<float>> reshape(const std::vector<float> &flat_matrix, const size_t &num_columns)
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

void bench(const std::vector<std::vector<float>> &X_train,
           const std::vector<float> &y_train,
           const std::vector<std::vector<float>> &X_test,
           const int num_iterations,
           const bool use_gpu,
           const bool verbose_gpu = false)
{
    std::vector<double> train_times, predict_times;

    std::cout << "\n===== STARTING BENCHMARK =====\n" << std::endl;
    
    if (use_gpu) {
        if (!verbose_gpu) {
            std::cout << "\033[1;32m[INFO] Running with GPU acceleration enabled\033[0m" << std::endl;
            std::cout << "GPU monitoring has been streamlined to show metrics only at critical points:\n" << std::endl;
            std::cout << "  * Initial GPU state" << std::endl;
            std::cout << "  * First data transfer" << std::endl;
            std::cout << "  * Progress indicators at 1000-item intervals" << std::endl; 
            std::cout << "  * Final summary with performance statistics" << std::endl;
            std::cout << "\nThis minimizes output while still providing essential GPU usage information." << std::endl;
            std::cout << "For verbose GPU monitoring, run with --verbose-gpu or -v flag.\n" << std::endl;
        } else {
            std::cout << "\033[1;32m[INFO] Running with GPU acceleration and VERBOSE monitoring enabled\033[0m" << std::endl;
            std::cout << "Full GPU metrics will be shown for each benchmark iteration." << std::endl;
            std::cout << "This may generate a lot of output, but provides the most detailed information.\n" << std::endl;
        }
    } else {
        std::cout << "\033[1;33m[INFO] Running with CPU only (no GPU acceleration)\033[0m" << std::endl;
        std::cout << "To enable GPU acceleration, run with --gpu flag\n" << std::endl;
    }

    for (int i = 0; i < num_iterations; i++)
    {
        if (i % 100 == 0 && num_iterations >= 100) {
            std::cout << "Progress: " << i << "/" << num_iterations << " iterations completed" << std::endl;
        }
        
        // In verbose mode, reset GPU metrics tracking between iterations
        if (use_gpu && verbose_gpu && i > 0) {
            // This will allow GPU metrics to be shown for each iteration
            extern void resetGPUMetricsTracking();
            resetGPUMetricsTracking();
        }
        
        // Start timing training
        auto start_train = std::chrono::high_resolution_clock::now();
        KNN model(X_train, y_train, 0, use_gpu); // Pass use_gpu flag
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
        
        // In verbose mode, add a separator between iterations
        if (use_gpu && verbose_gpu && i < num_iterations - 1) {
            std::cout << "\n---------- End of iteration " << (i+1) << "/" << num_iterations << " ----------\n" << std::endl;
        }
    }

    // Compute mean and standard deviation
    auto mean_std = [](const std::vector<double> &times)
    {
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = std::accumulate(times.begin(), times.end(), 0.0,
                                          [mean](double acc, double x)
                                          { return acc + (x - mean) * (x - mean); }) /
                          times.size();
        double std_dev = std::sqrt(variance);
        return std::make_pair(mean, std_dev);
    };

    auto [train_mean, train_std] = mean_std(train_times);
    auto [pred_mean, pred_std] = mean_std(predict_times);

    // Print formatted results
    std::cout << "\n===== Experiment Parameters =====" << std::endl;
    std::cout << std::left << std::setw(24) << "Number of samples" << ": " << X_train.size() + X_test.size() << std::endl;
    std::cout << std::left << std::setw(24) << "Number of features" << ": " << X_train[0].size() << std::endl;
    std::cout << std::left << std::setw(24) << "Number of iterations" << ": " << num_iterations << std::endl;
    std::cout << std::left << std::setw(24) << "Algorithm" << ": brute" << std::endl;
    
    if (use_gpu) {
        std::cout << std::left << std::setw(24) << "Using GPU" << ": " << "\033[1;32mYes\033[0m" << std::endl;
        if (verbose_gpu) {
            std::cout << std::left << std::setw(24) << "GPU Monitoring" << ": " << "\033[1;32mVerbose\033[0m" << std::endl;
        }
    } else {
        std::cout << std::left << std::setw(24) << "Using GPU" << ": " << "\033[1;33mNo\033[0m" << std::endl;
    }

    std::cout << "\n===== Performance Statistics =====" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(24) << "Fit Time (Mean)" << ": " << std::setw(8) << train_mean << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Fit Time (Std Dev)" << ": " << std::setw(8) << train_std << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Mean)" << ": " << std::setw(8) << pred_mean << " ms" << std::endl;
    std::cout << std::left << std::setw(24) << "Inference Time (Std Dev)" << ": " << std::setw(8) << pred_std << " ms" << std::endl;
    
    // Add speedup analysis if both CPU and GPU are benchmarked
    std::cout << "\nNote: For detailed GPU metrics, check the output above." << std::endl;
}

int main(int argc, char** argv)
{
    bool use_gpu = false;
    bool verbose_gpu_monitoring = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0 || strcmp(argv[i], "-g") == 0) {
            use_gpu = true;
        }
        else if (strcmp(argv[i], "--verbose-gpu") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose_gpu_monitoring = true;
        }
    }

    const size_t num_features = 16;
    const size_t num_samples = 10000;
    const int num_iterations = 500; // Number of iterations for benchmarking

    std::string X_path = "../data/X.npy";
    std::string y_path = "../data/y.npy";

    std::cout << "Loading dataset from " << X_path << " and " << y_path << std::endl;
    
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

    std::cout << "Dataset loaded. Training set: " << X_train.size() << " samples, Test set: " << X_test.size() << " samples" << std::endl;
    
    if (use_gpu && verbose_gpu_monitoring) {
        std::cout << "\nVerbose GPU monitoring enabled. GPU metrics will be displayed for each benchmark iteration." << std::endl;
        // Update the bench function to inform about verbose mode
        std::cout << "Run with --gpu only (without -v/--verbose-gpu) for minimal GPU output.\n" << std::endl;
    }
    
    // Call benchmark function
    bench(X_train, y_train, X_test, num_iterations, use_gpu, verbose_gpu_monitoring);

    std::cout << "\nBenchmark completed." << std::endl;
    
    return 0;
}
