import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from loguru import logger
from collections import Counter
from time import time

def main():
    n_samples = int(1e6)
    n_features = 100
    n_informative = 1
    n_redundant = 1
    n_classes = 2
    n_clusters_pre_class = 1
    k = 100  # Number of neighbors

    # Generate dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_pre_class
    )

    np.random.seed(42)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001, shuffle=False)
    
    logger.info("Train and Validation sets created")

    train_start = time()
    # Precompute squared norms of training samples
    train_norms = np.sum(X_train**2, axis=1)
    train_time = (time() - train_start) / 1e-6
    logger.info(f"Training complete in {train_time:.3f} us. {train_time / len(y_train) / 1e-3:.3f} ns per sample")

    val_start = time()
    preds = []
    for vsample, vtarget in zip(X_val, y_val):
        # Compute squared distances using vectorized operations
        distances = train_norms - 2 * np.dot(X_train, vsample) + np.sum(vsample**2)
        
        # Get the indices of the k nearest neighbors
        nearest_indices = np.argpartition(distances, k)[:k]
        
        # Get the labels of the nearest neighbors
        nearest_labels = y_train[nearest_indices]
        
        # Predict the majority class
        pred = Counter(nearest_labels).most_common(1)[0][0]
        preds.append((vtarget, pred))
    val_time = (time() - val_start) / 1e-3
    logger.info(f"Validation complete in {val_time:.3f} ms. {val_time / len(y_val) / 1e-3:.3f} us per sample")
    
    accuracy = np.mean([v == t for v, t in preds])
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()