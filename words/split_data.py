import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X: np.ndarray, y: np.ndarray, seed: int = 42):
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
        )
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, stratify=None
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
 
