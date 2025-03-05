import numpy as np
import os
import sys
import platform
import multiprocessing

def print_threading_info():
    print("===== System and Python Environment =====")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Executable: {sys.executable}")
    
    print("\n===== Thread and CPU Information =====")
    print(f"CPU Count: {multiprocessing.cpu_count()}")
    
    print("\n===== NumPy Threading Configuration =====")
    try:
        import numpy as np
        print("NumPy Config:")
        print(np.__config__.show())
    except Exception as e:
        print(f"Could not retrieve NumPy config: {e}")
    
    print("\n===== Environment Variables =====")
    thread_env_vars = [
        'OMP_NUM_THREADS', 
        'MKL_NUM_THREADS', 
        'OPENBLAS_NUM_THREADS', 
        'NUMEXPR_NUM_THREADS', 
        'VECLIB_MAXIMUM_THREADS'
    ]
    for var in thread_env_vars:
        print(f"{var}: {os.environ.get(var, 'Not Set')}")
    
    print("\n===== Scikit-learn Backend =====")
    try:
        import threadpoolctl
        print("Current library thread limits:")
        print(threadpoolctl.threadpool_info())
    except ImportError:
        print("threadpoolctl not installed. Install with: pip install threadpoolctl")

# Run the diagnostic function
print_threading_info()

# Comprehensive threading control script
import numpy as np
import os
import ctypes
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier

# Attempt to disable threading at multiple levels
def minimize_threading():
    # Environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # Try to disable OpenMP
    try:
        import ctypes
        libomp = ctypes.CDLL('libgomp.so.1')  # Linux
        libomp.omp_set_num_threads(1)
    except:
        try:
            libomp = ctypes.CDLL('libiomp5.dylib')  # macOS
            libomp.omp_set_num_threads(1)
        except:
            pass

    # Scikit-learn configuration
    try:
        import sklearn
        sklearn.set_config(n_jobs=1)
    except ImportError:
        pass

    # Threadpoolctl if available
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1, user_api='blas')
        threadpoolctl.threadpool_limits(limits=1, user_api='openmp')
    except ImportError:
        pass

# Run threading minimization
minimize_threading()

def execute_experiment(X, y, iters=3):
    from time import perf_counter_ns
    
    fit_perfs = []
    inf_perfs = []
    
    for _ in range(iters):
        # Extremely strict single-thread configuration
        model = KNeighborsClassifier(
            algorithm='brute', 
            n_jobs=1,  # Explicitly set to 1
            # Additional parameters if supported
            leaf_size=30,  # Avoid auto-threading optimizations
        )
        
        fit_perf_start = perf_counter_ns()
        model.fit(X, y)
        fit_perf_end = perf_counter_ns()
        fit_perfs.append(fit_perf_end - fit_perf_start)
        
        inf_perf_start = perf_counter_ns()
        model.predict(X)
        inf_perf_end = perf_counter_ns()
        inf_perfs.append(inf_perf_end - inf_perf_start)
    
    return fit_perfs, inf_perfs

# Actual experiment execution
if __name__ == "__main__":
    # Load data
    X = np.load("../data/X.npy")
    y = np.load("../data/y.npy")
    
    # Run experiment with extreme threading control
    fit_perfs, inf_perfs = execute_experiment(X, y)
    
    # Verification of single-threading
    print("\n===== Thread Usage Verification =====")
    print(f"Number of threads used in NumPy: {np.show_config()}")
    
    try:
        import threadpoolctl
        print("Current thread limits:")
        print(threadpoolctl.threadpool_info())
    except ImportError:
        pass