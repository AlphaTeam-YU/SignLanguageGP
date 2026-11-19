"""Joint-angle features for hand landmarks (letters part).

Input: wrist-centred, size-normalised landmarks of shape (N, 63), laid out as
(x0, y0, z0, x1, y1, z1, ...) in MediaPipe Hands order.
Angles are invariant to translation, rotation and uniform scaling.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

Triplet = Tuple[int, int, int]  # (a, vertex b, c): angle at b between b->a and b->c


class JointAngleFeatures:
    NUM_LANDMARKS = 21
    NUM_COORDS = 63
    WRIST = 0
    FINGERS = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    @classmethod
    def default_triplets(cls) -> List[Tuple[str, Triplet]]:
        """26 angles recovered from the dataset CSV (exact match, 3D, MAE=0).
        (a, vertex, c): angle at vertex between vertex->a and vertex->c.
        """
        t = [
            (0, 1, 5), (0, 1, 17), (0, 1, 2), (0, 5, 17), (0, 5, 6), (0, 5, 9),
            (0, 17, 13), (0, 17, 18), (1, 2, 3), (2, 3, 4), (5, 6, 9), (5, 6, 7),
            (5, 9, 10), (5, 9, 13), (6, 7, 8), (9, 10, 13), (9, 10, 11),
            (9, 13, 14), (9, 13, 17), (10, 11, 12), (13, 14, 17), (13, 14, 15),
            (13, 17, 18), (14, 15, 16), (17, 18, 19), (18, 19, 20),
        ]

        return [(f"ang_{a}_{b}_{c}", (a, b, c)) for a, b, c in t]

    def __init__(self, triplets: Optional[Sequence[Tuple[str, Triplet]]] = None,
                 eps: float = 1e-8, use_2d: bool = False):
        named = list(triplets) if triplets is not None else self.default_triplets()
        self.names = [n for n, _ in named]
        idx = np.array([t for _, t in named], dtype=np.int64)
        self._a, self._b, self._c = idx[:, 0], idx[:, 1], idx[:, 2]
        self.eps = eps
        self.use_2d = use_2d  # True: angles from (x, y) only

    @property
    def num_angles(self) -> int:
        return len(self.names)

    @property
    def feature_names(self) -> List[str]:
        coords = [f"{ax}{i}" for i in range(self.NUM_LANDMARKS) for ax in "xyz"]
        return coords + [f"angle_{n}" for n in self.names]

    def _to_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.NUM_COORDS:
            raise ValueError(f"Expected shape (N, 63), got {X.shape}")
        return X.reshape(-1, self.NUM_LANDMARKS, 3).astype(np.float64)

    def compute_angles(self, X: np.ndarray) -> np.ndarray:
        """Angles in degrees, shape (N, num_angles), float32."""
        pts = self._to_points(X)
        if self.use_2d:
            pts = pts[..., :2]
        v1 = pts[:, self._a, :] - pts[:, self._b, :]
        v2 = pts[:, self._c, :] - pts[:, self._b, :]
        dot = np.sum(v1 * v2, axis=-1)
        norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
        cos = np.clip(dot / np.maximum(norm, self.eps), -1.0, 1.0)
        return np.degrees(np.arccos(cos)).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Concatenate the 63 normalised coordinates with the joint angles."""
        X = np.asarray(X, dtype=np.float32)
        return np.concatenate([X, self.compute_angles(X)], axis=1)
