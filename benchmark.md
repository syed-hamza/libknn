### Note
- As of 5th march 2025, C++ implementation does not support threaded execution. Results will be updates once multithreaded feature is added to C++
- The difference in performance metric is dude to the different sizes of training and testing samples used in both implementations

## Python 
```

===== Running Experiment with n_jobs=1 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 1

===== Performance Statistics =====
Fit Time (Mean)          :   1.755 ms
Fit Time (Std Dev)       : 202.006 ms
Inference Time (Mean)    : 536.681 ms
Inference Time (Std Dev) :  26.802 ms
Fit Time per Sample      : 87.756983 ns
Inference Time per Sample: 26.834027 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=2 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 2

===== Performance Statistics =====
Fit Time (Mean)          :   1.526 ms
Fit Time (Std Dev)       : 130.945 ms
Inference Time (Mean)    : 507.667 ms
Inference Time (Std Dev) :   4.655 ms
Fit Time per Sample      : 76.284133 ns
Inference Time per Sample: 25.383325 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=3 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 3

===== Performance Statistics =====
Fit Time (Mean)          :   1.589 ms
Fit Time (Std Dev)       : 142.899 ms
Inference Time (Mean)    : 607.002 ms
Inference Time (Std Dev) :  23.389 ms
Fit Time per Sample      : 79.458017 ns
Inference Time per Sample: 30.350123 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=4 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 4

===== Performance Statistics =====
Fit Time (Mean)          :   1.469 ms
Fit Time (Std Dev)       :  63.351 ms
Inference Time (Mean)    : 693.566 ms
Inference Time (Std Dev) :   7.360 ms
Fit Time per Sample      : 73.474100 ns
Inference Time per Sample: 34.678314 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=5 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 5

===== Performance Statistics =====
Fit Time (Mean)          :   1.469 ms
Fit Time (Std Dev)       :  99.362 ms
Inference Time (Mean)    : 594.543 ms
Inference Time (Std Dev) :  80.065 ms
Fit Time per Sample      : 73.438700 ns
Inference Time per Sample: 29.727136 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=6 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 6

===== Performance Statistics =====
Fit Time (Mean)          :   1.562 ms
Fit Time (Std Dev)       : 190.691 ms
Inference Time (Mean)    : 535.427 ms
Inference Time (Std Dev) :  20.771 ms
Fit Time per Sample      : 78.094583 ns
Inference Time per Sample: 26.771370 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=7 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 7

===== Performance Statistics =====
Fit Time (Mean)          :   1.571 ms
Fit Time (Std Dev)       : 190.969 ms
Inference Time (Mean)    : 668.298 ms
Inference Time (Std Dev) :  30.806 ms
Fit Time per Sample      : 78.557433 ns
Inference Time per Sample: 33.414925 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


===== Running Experiment with n_jobs=8 =====

===== Experiment Parameters =====
Number of samples        : 20000
Number of features       : 100
Number of iterations     : 3
Algorithm                : brute
n_jobs                   : 8

===== Performance Statistics =====
Fit Time (Mean)          :   1.460 ms
Fit Time (Std Dev)       :  42.772 ms
Inference Time (Mean)    : 522.323 ms
Inference Time (Std Dev) :   4.734 ms
Fit Time per Sample      : 73.017533 ns
Inference Time per Sample: 26.116126 µs

===== Performance Metrics =====
              precision    recall  f1-score   support

         0.0       0.92      0.88      0.90     10011
         1.0       0.89      0.92      0.90      9989

    accuracy                           0.90     20000
   macro avg       0.90      0.90      0.90     20000
weighted avg       0.90      0.90      0.90     20000


```

### Note
- As of 5th march, C++ implementation does not support threaded execution. Results will be updates once multithreaded feature is added to C++
- The difference in performance metric is dude to the size of training and testing samples used in both implementations

## C++
```
===== Experiment Parameters =====
Number of Training Samples    : 10000
Number of Testing Samples     : 10000
Total Number of Samples       : 20000
Number of features            : 100
Number of iterations          : 5
Algorithm                     : brute
===== Performance Statistics =====
Fit Time (Mean)         : 2.555    ms
Fit Time (Std Dev)      : 87.109   ms
Inference Time (Mean)   : 2485.769 ms
Inference Time (Std Dev): 32.490   ms
Train Time per Sample   : 255.504  ns/sample
Predict Time per Sample : 248.577  µs/sample

Classification Report:
--------------------------------------
Class | Precision | Recall | F1-score
--------------------------------------
1.0000     | 0.9140     | 0.9900   | 0.9505
0.0000     | 0.9891     | 0.9067   | 0.9461
--------------------------------------
Macro Avg    | 0.9516   | 0.9484   | 0.9483
Weighted Avg | 0.9515   | 0.9484   | 0.9483
```