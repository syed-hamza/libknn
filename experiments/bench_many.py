import numpy as np
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from loguru import logger


def run_experiment(params):
    """Run a single KNN experiment with given parameters"""
    m, n, algorithm = params
    logger.info(f"Started experiment with m: {m}, n: {n}, algorithm: {algorithm}.")
    # Use numpy's random number generator directly - it's faster
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 1, (n, m))
    y = rng.randint(0, 2, n)  # Faster than list comprehension

    model = KNeighborsClassifier(algorithm=algorithm, n_jobs=1)

    # Use perf_counter for more precise timing
    fit_start = perf_counter()
    model.fit(X, y)
    fit_time = perf_counter() - fit_start

    inf_start = perf_counter()
    model.predict(X)
    inf_time = perf_counter() - inf_start
    logger.info(f"Ended experiment with m: {m}, n: {n}, algorithm: {algorithm}.")

    return fit_time, inf_time


def run_all_experiments(M, N, algorithms, n_workers=4):
    """Run all experiments in parallel for each algorithm"""
    fit_times = {alg: np.empty((len(M), len(N))) for alg in algorithms}
    inf_times = {alg: np.empty((len(M), len(N))) for alg in algorithms}

    experiments = [
        (m, n, alg)
        for alg in algorithms
        for i, m in enumerate(M)
        for j, n in enumerate(N)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(run_experiment, experiments))

    idx = 0
    for alg in algorithms:
        for i, m in enumerate(M):
            for j, n in enumerate(N):
                fit_times[alg][i, j], inf_times[alg][i, j] = results[idx]
                idx += 1

    return fit_times, inf_times


def plot_algorithm_results(M, N, fit_times, inf_times, algorithm):
    """Create detailed visualization for a single algorithm"""
    fig = plt.figure(figsize=(15, 12))

    # Common plotting parameters
    plot_params = {"cmap": "viridis", "aspect": "auto", "interpolation": "nearest"}

    # Create a 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 0.2], height_ratios=[1, 1])

    # Fitting times heatmap
    ax_fit = fig.add_subplot(gs[0, 0])
    im1 = ax_fit.imshow(fit_times[algorithm], **plot_params)
    ax_fit.set_title(f"{algorithm.upper()} Fitting Times (seconds)", pad=20)
    ax_fit.set_xticks(range(0, len(N), len(N) // 5))
    ax_fit.set_yticks(range(0, len(M), len(M) // 5))
    ax_fit.set_xticklabels([f"n={N[i]}" for i in range(0, len(N), len(N) // 5)])
    ax_fit.set_yticklabels([f"m={M[i]}" for i in range(0, len(M), len(M) // 5)])
    ax_fit.set_xlabel("Number of Samples (N)")
    ax_fit.set_ylabel("Number of Features (M)")

    # Colorbar for fitting times
    cax_fit = fig.add_subplot(gs[0, 1])
    plt.colorbar(im1, cax=cax_fit, label="Seconds")

    # Inference times heatmap
    ax_inf = fig.add_subplot(gs[1, 0])
    im2 = ax_inf.imshow(inf_times[algorithm], **plot_params)
    ax_inf.set_title(f"{algorithm.upper()} Inference Times (seconds)", pad=20)
    ax_inf.set_xticks(range(0, len(N), len(N) // 5))
    ax_inf.set_yticks(range(0, len(M), len(M) // 5))
    ax_inf.set_xticklabels([f"n={N[i]}" for i in range(0, len(N), len(N) // 5)])
    ax_inf.set_yticklabels([f"m={M[i]}" for i in range(0, len(M), len(M) // 5)])
    ax_inf.set_xlabel("Number of Samples (N)")
    ax_inf.set_ylabel("Number of Features (M)")

    # Colorbar for inference times
    cax_inf = fig.add_subplot(gs[1, 1])
    plt.colorbar(im2, cax=cax_inf, label="Seconds")

    plt.suptitle(
        f"Performance Analysis of {algorithm.upper()} Algorithm", y=0.95, fontsize=16
    )
    plt.tight_layout()
    return fig


def plot_all_algorithms(M, N, fit_times, inf_times, algorithms):
    """Plot separate figures for each algorithm"""
    figures = []
    for algorithm in algorithms:
        fig = plot_algorithm_results(M, N, fit_times, inf_times, algorithm)
        figures.append(fig)
    return figures


# Example usage
if __name__ == "__main__":
    M = np.arange(1, 51, 1)  # Feature dimensions
    N = np.arange(100, 2100, 100)  # Sample sizes
    algorithms = ["ball_tree", "kd_tree", "brute"]  # Different KNN algorithms

    fit_times, inf_times = run_all_experiments(M, N, algorithms)
    figures = plot_all_algorithms(M, N, fit_times, inf_times, algorithms)

    # Show all figures
    plt.show()
