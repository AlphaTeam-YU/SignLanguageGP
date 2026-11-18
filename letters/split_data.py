"""Group-aware stratified train/val/test split and train-only standardization (letters part)."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split


class FeatureStandardizer:
    """Z-score scaler fitted on the TRAIN split only (avoids data leakage)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "FeatureStandardizer":
        self.mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        # Constant features (e.g. the wrist, always 0) keep std=1 to avoid division by zero
        self.std = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "FeatureStandardizer":
        data = np.load(path)
        obj = cls()
        obj.mean, obj.std = data["mean"], data["std"]
        return obj


class LetterDataSplitter:
    """70/15/15 split that keeps duplicate / related samples together (no leakage).

    Rows that share identical landmarks (and, optionally, the same label+file number)
    are merged into one group; a whole group always goes to ONE split. Groups are
    assigned per class, so every class keeps ~70/15/15.
    """

    def __init__(self, train_size: float = 0.70, val_size: float = 0.15,
                 test_size: float = 0.15, seed: int = 42):
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Split sizes must sum to 1.0")
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        self.seed = seed

    # ---------- groups ----------
    @staticmethod
    def _link(values: np.ndarray):
        """Edges connecting every row to the first row having the same value."""
        _, first, inv = np.unique(values, return_index=True, return_inverse=True)
        return np.arange(len(values)), first[inv.ravel()]

    @classmethod
    def build_groups(cls, coords: np.ndarray, name_keys: Optional[np.ndarray] = None) -> np.ndarray:
        """Group id per row: identical landmarks (+ identical name_keys) share a group."""
        n = len(coords)
        df = pd.DataFrame(np.asarray(coords))
        dup_id = df.groupby(list(df.columns), sort=False).ngroup().to_numpy()
        rows, cols = cls._link(dup_id)
        if name_keys is not None:
            key_id = pd.factorize(pd.Series(name_keys))[0]
            r2, c2 = cls._link(key_id)
            rows, cols = np.concatenate([rows, r2]), np.concatenate([cols, c2])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, groups = connected_components(graph, directed=False)
        return groups

    # ---------- splits ----------
    def split_indices(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Plain stratified split (ignores duplicates). Kept for comparison."""
        idx = np.arange(len(y))
        train_idx, rest_idx = train_test_split(
            idx, test_size=self.val_size + self.test_size,
            stratify=y, random_state=self.seed, shuffle=True)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=self.test_size / (self.val_size + self.test_size),
            stratify=y[rest_idx], random_state=self.seed, shuffle=True)
        return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}

    def split_indices_grouped(self, y: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
        """Group-aware stratified split: a group is never divided between splits."""
        rng = np.random.default_rng(self.seed)
        fractions = {"train": self.train_size, "val": self.val_size, "test": self.test_size}
        parts: Dict[str, List[np.ndarray]] = {k: [] for k in fractions}
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            ug, inv = np.unique(groups[idx_c], return_inverse=True)
            inv = inv.ravel()
            sizes = np.bincount(inv)
            order = rng.permutation(len(ug))
            order = order[np.argsort(-sizes[order], kind="stable")]   # big groups first
            target = {k: f * len(idx_c) for k, f in fractions.items()}
            count = {k: 0 for k in fractions}
            assign = np.empty(len(ug), dtype="<U5")
            for g in order:
                k = max(fractions, key=lambda s: target[s] - count[s])
                assign[g] = k
                count[k] += sizes[g]
            for k in fractions:
                parts[k].append(idx_c[assign[inv] == k])
        return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}

    @staticmethod
    def apply(X: np.ndarray, y: np.ndarray,
              indices: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {name: (X[i], y[i]) for name, i in indices.items()}

    @staticmethod
    def save(out_dir: str, data, indices, scaler: FeatureStandardizer,
             feature_names: List[str]) -> None:
        """Save raw features, standardized features, labels, indices and the scaler."""
        os.makedirs(out_dir, exist_ok=True)
        for name, (Xk, yk) in data.items():
            np.save(os.path.join(out_dir, f"X_{name}.npy"), Xk.astype(np.float32))
            np.save(os.path.join(out_dir, f"X_{name}_std.npy"), scaler.transform(Xk))
            np.save(os.path.join(out_dir, f"y_{name}.npy"), yk)
        np.savez(os.path.join(out_dir, "split_indices.npz"), **indices)
        scaler.save(os.path.join(out_dir, "scaler.npz"))
        with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f)
