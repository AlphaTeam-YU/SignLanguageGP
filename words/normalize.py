from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml


class WordNormalizer:
    """Shoulder-centered and shoulder-width scale normalizer for holistic word sequences.

    Matches WordLandmarkExtractor output shape: (num_frames, 378) or (N, num_frames, 378)
    Layout per frame (378 features):
      - Pose:       33 landmarks * 4 (x, y, z, visibility) -> [0:132]
      - Face:       40 landmarks * 3 (x, y, z)             -> [132:252]
      - Left hand:  21 landmarks * 3 (x, y, z)             -> [252:315]
      - Right hand: 21 landmarks * 3 (x, y, z)             -> [315:378]
    """

    def __init__(self, config_path: str = "configs/words.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.norm_cfg = self.config.get("normalization", {})
        self.left_shoulder_idx = int(self.norm_cfg.get("left_shoulder_idx", 11))
        self.right_shoulder_idx = int(self.norm_cfg.get("right_shoulder_idx", 12))
        self.eps = float(self.norm_cfg.get("eps", 1e-6))
        self.use_z_in_scale = bool(self.norm_cfg.get("use_z_in_scale", False))
        self.min_visibility = float(self.norm_cfg.get("min_visibility", 0.0))
        fw = self.norm_cfg.get("frame_width")
        fh = self.norm_cfg.get("frame_height")
        self.default_frame_size: Optional[Tuple[float, float]] = (
            (float(fw), float(fh)) if fw and fh else None
        )

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _missing_mask(block: np.ndarray) -> np.ndarray:
        """(T, K, 3) -> (T, K, 1) bool, True where the point is all-zero (missing)."""
        return np.all(np.isclose(block, 0.0, atol=1e-7), axis=-1, keepdims=True)

    @staticmethod
    def _normalize_xyz_block(
        block: np.ndarray, center: np.ndarray, scale: np.ndarray, eps: float
    ) -> np.ndarray:
        """Centers and scales (T, K, 3) landmarks while preserving missing (all-zero) points."""
        missing_mask = WordNormalizer._missing_mask(block)
        normed = (block - center[:, None, :]) / (scale[:, None, :] + eps)
        return np.where(missing_mask, 0.0, normed)

    @staticmethod
    def _forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Forward-fills (T, D) values at invalid frames with the last valid one."""
        if not np.any(valid):
            return np.zeros_like(values)
        T = values.shape[0]
        idx = np.where(valid, np.arange(T), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = int(np.argmax(valid))
        return values[idx]

    @staticmethod
    def _apply_frame_size(block: np.ndarray, frame_size: Tuple[float, float]) -> np.ndarray:
        """Converts MediaPipe-normalized coords to a common aspect ratio."""
        w, h = frame_size
        factors = np.array([w, h, w], dtype=np.float32)
        return block * factors

    def normalize_sequence(
        self, seq: np.ndarray, frame_size: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Normalizes a single video sequence of shape (T, 378)."""
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[1] != 378:
            raise ValueError(f"Expected sequence shape (T, 378), got {seq.shape}")

        frame_size = frame_size or self.default_frame_size
        T = seq.shape[0]

        pose = seq[:, :132].reshape(T, 33, 4).copy()
        face = seq[:, 132:252].reshape(T, 40, 3).copy()
        left_hand = seq[:, 252:315].reshape(T, 21, 3).copy()
        right_hand = seq[:, 315:378].reshape(T, 21, 3).copy()

        pose_xyz = pose[..., :3]
        pose_vis = pose[..., 3:4]

        if self.min_visibility > 0.0:
            low_vis = pose_vis < self.min_visibility
            pose_xyz = np.where(low_vis, 0.0, pose_xyz)

        if frame_size is not None:
            pose_xyz = self._apply_frame_size(pose_xyz, frame_size)
            face = self._apply_frame_size(face, frame_size)
            left_hand = self._apply_frame_size(left_hand, frame_size)
            right_hand = self._apply_frame_size(right_hand, frame_size)

        l_shoulder = pose_xyz[:, self.left_shoulder_idx, :]
        r_shoulder = pose_xyz[:, self.right_shoulder_idx, :]

        shoulders_present = ~(
            self._missing_mask(l_shoulder[:, None, :])[:, 0, 0]
            | self._missing_mask(r_shoulder[:, None, :])[:, 0, 0]
        )

        center = 0.5 * (l_shoulder + r_shoulder)

        diff = l_shoulder - r_shoulder
        if not self.use_z_in_scale:
            diff = diff[:, :2]
        frame_scale = np.linalg.norm(diff, axis=1, keepdims=True)

        valid_frames = shoulders_present & (frame_scale[:, 0] > self.eps)

        center = self._forward_fill(center, valid_frames)
        if np.any(valid_frames):
            mean_scale = float(np.mean(frame_scale[valid_frames]))
            scale = np.where(valid_frames[:, None], frame_scale, mean_scale)
        else:
            scale = np.ones_like(frame_scale)

        pose_xyz_norm = self._normalize_xyz_block(pose_xyz, center, scale, self.eps)
        face_norm = self._normalize_xyz_block(face, center, scale, self.eps)
        lh_norm = self._normalize_xyz_block(left_hand, center, scale, self.eps)
        rh_norm = self._normalize_xyz_block(right_hand, center, scale, self.eps)

        pose_norm = np.concatenate([pose_xyz_norm, pose_vis], axis=-1)

        return np.concatenate(
            [
                pose_norm.reshape(T, 132),
                face_norm.reshape(T, 120),
                lh_norm.reshape(T, 63),
                rh_norm.reshape(T, 63),
            ],
            axis=1,
        ).astype(np.float32)

    def normalize_batch(
        self, X: np.ndarray, frame_size: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Normalizes a batch of sequences of shape (N, T, 378)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[2] != 378:
            raise ValueError(f"Expected batch shape (N, T, 378), got {X.shape}")
        return np.stack([self.normalize_sequence(seq, frame_size) for seq in X], axis=0)
