import numpy as np
from time import perf_counter_ns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from loguru import logger
import os

# Explicitly set threading parameters
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

def convert_time(ns):
    units = ["ns", "µs", "ms", "s"]
    scale = [1, 1e3, 1e6, 1e9]
    
    for unit, factor in zip(units, scale):
        if ns < factor * 1000:
            return ns / factor, unit
    
    return ns / 1e9, "s"

def execute_experiment(params):
    n_jobs = params["n_jobs"]
    algorithm = params['algorithm']
    iters = params['iters']

    X = np.load("../data/X.npy")
    y = np.load("../data/y.npy")

    m, n = X.shape

    fit_perfs = []
    inf_perfs = []
    
    for _ in range(iters):
        model = KNeighborsClassifier(algorithm=algorithm, n_jobs=n_jobs)
        
        fit_perf_start = perf_counter_ns()
        model.fit(X, y)
        fit_perf_end = perf_counter_ns()
        fit_perfs.append(fit_perf_end - fit_perf_start)
        
        inf_perf_start = perf_counter_ns()
        model.predict(X)
        inf_perf_end = perf_counter_ns()
        inf_perfs.append(inf_perf_end - inf_perf_start)

    mean_fit, fit_unit = convert_time(np.mean(fit_perfs))
    std_fit, _ = convert_time(np.std(fit_perfs))
    mean_inf, inf_unit = convert_time(np.mean(inf_perfs))
    std_inf, _ = convert_time(np.std(inf_perfs))

    fit_time_per_sample, fit_sample_unit = convert_time(np.mean(fit_perfs) / m)
    inf_time_per_sample, inf_sample_unit = convert_time(np.mean(inf_perfs) / m)

    report = classification_report(y, model.predict(X))

    print("\n===== Experiment Parameters =====")
    print(f"{'Number of samples':<25}: {m}")
    print(f"{'Number of features':<25}: {n}")
    print(f"{'Number of iterations':<25}: {iters}")
    print(f"{'Algorithm':<25}: {algorithm}")
    print(f"{'n_jobs':<25}: {n_jobs}")

    print("\n===== Performance Statistics =====")
    print(f"{'Fit Time (Mean)':<25}: {mean_fit:7.3f} {fit_unit}")
    print(f"{'Fit Time (Std Dev)':<25}: {std_fit:7.3f} {fit_unit}")
    print(f"{'Inference Time (Mean)':<25}: {mean_inf:7.3f} {inf_unit}")
    print(f"{'Inference Time (Std Dev)':<25}: {std_inf:7.3f} {inf_unit}")
    print(f"{'Fit Time per Sample':<25}: {fit_time_per_sample:7.6f} {fit_sample_unit}")
    print(f"{'Inference Time per Sample':<25}: {inf_time_per_sample:7.6f} {inf_sample_unit}")

    print("\n===== Performance Metrics =====")
    print(report)

if __name__ == "__main__":
    for n_jobs in range(1, 2):
        print(f"\n===== Running Experiment with n_jobs={n_jobs} =====")
        params = {
            "n_jobs": n_jobs,
            "algorithm": "brute",
            "iters": 3
        }
        execute_experiment(params)