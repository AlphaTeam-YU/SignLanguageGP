"""Landmark augmentation for the letters part (light rotation + optional jitter).

Works on wrist-centred, hand-size-normalised coordinates (N, 63).
Rotation is done around the wrist (origin), so the wrist stays at 0 and the
hand size (wrist -> middle MCP) stays 1.0. NO scale augmentation: the real
data always has hand size = 1.0, so scaling would create unrealistic inputs.
Joint angles are invariant to rotation, but are recomputed anyway in
augment_features() to keep the pipeline consistent.
"""
from typing import Optional

import numpy as np


class LandmarkAugmenter:
    MIDDLE_MCP = 9

    def __init__(self, rot_z_deg: float = 15.0, rot_xy_deg: float = 5.0,
                 jitter_std: float = 0.0, seed: Optional[int] = None):
        self.rot_z_deg = rot_z_deg      # in-plane rotation limit (degrees)
        self.rot_xy_deg = rot_xy_deg    # small out-of-plane tilt limit (degrees)
        self.jitter_std = jitter_std    # optional Gaussian noise (0 = off)
        self.rng = np.random.default_rng(seed)

    def _rotation_matrices(self, n: int) -> np.ndarray:
        """Random R = Rz @ Ry @ Rx for each sample, shape (n, 3, 3)."""
        ax = np.deg2rad(self.rng.uniform(-self.rot_xy_deg, self.rot_xy_deg, n))
        ay = np.deg2rad(self.rng.uniform(-self.rot_xy_deg, self.rot_xy_deg, n))
        az = np.deg2rad(self.rng.uniform(-self.rot_z_deg, self.rot_z_deg, n))
        cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
        R = np.zeros((n, 3, 3), dtype=np.float64)
        R[:, 0, 0] = cz * cy
        R[:, 0, 1] = cz * sy * sx - sz * cx
        R[:, 0, 2] = cz * sy * cx + sz * sx
        R[:, 1, 0] = sz * cy
        R[:, 1, 1] = sz * sy * sx + cz * cx
        R[:, 1, 2] = sz * sy * cx - cz * sx
        R[:, 2, 0] = -sy
        R[:, 2, 1] = cy * sx
        R[:, 2, 2] = cy * cx
        return R

    def augment(self, X: np.ndarray) -> np.ndarray:
        """Return an augmented copy of X (N, 63); labels stay unchanged."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 63:
            raise ValueError(f"Expected shape (N, 63), got {X.shape}")
        n = X.shape[0]
        pts = X.reshape(n, 21, 3)
        R = self._rotation_matrices(n)
        pts = np.einsum("nij,nkj->nki", R, pts)                       # rotate every point
        if self.jitter_std > 0:
            pts = pts + self.rng.normal(0.0, self.jitter_std, pts.shape)
            pts[:, 0, :] = 0.0                                        # keep wrist at origin
        # Re-normalise hand size to exactly 1.0 (same as C's preprocessing)
        size = np.linalg.norm(pts[:, self.MIDDLE_MCP, :], axis=1)
        pts = pts / np.maximum(size, 1e-8)[:, None, None]
        return pts.reshape(n, 63).astype(np.float32)

    def augment_features(self, X63: np.ndarray, featurizer, scaler) -> np.ndarray:
        """augment -> joint angles -> train-fitted standardization.

        Returns (N, 89) standardized features, ready to be added to the TRAIN set.
        Use on the train split only (never on val/test).
        """
        Xa = self.augment(X63)
        return scaler.transform(featurizer.transform(Xa))
