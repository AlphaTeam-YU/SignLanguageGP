"""Dataset loading, train-only light hand rotation augmentation, and tf.data pipelines."""

import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from letters.augment import LandmarkAugmenter
from letters.features import JointAngleFeatures
from letters.split_data import FeatureStandardizer

NUM_COORDS = 63


def _load_split_arrays(
    data_dir: str, split: str, scaler: FeatureStandardizer
) -> Tuple[np.ndarray, np.ndarray]:
  y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
  std_path = os.path.join(data_dir, f"X_{split}_std.npy")
  raw_path = os.path.join(data_dir, f"X_{split}.npy")

  if os.path.exists(std_path):
    X = np.load(std_path).astype(np.float32)
  elif os.path.exists(raw_path):
    X_raw = np.load(raw_path).astype(np.float32)
    X = scaler.transform(X_raw)
  else:
    raise FileNotFoundError(
        f"Missing feature file for split '{split}' in {data_dir}"
    )

  return X, y


def build_datasets(
    data_dir: str,
    batch_size: int = 64,
    seed: int = 42,
    aug_copies: int = 1,
    rot_z_deg: float = 15.0,
    rot_xy_deg: float = 5.0,
    shuffle_buffer: int = 10000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Build (train_ds, val_ds, test_ds) using pre-split data in `data_dir`.

    Light hand rotation augmentation is applied ONLY to the train split.
    Validation and test splits remain untouched.
    """
  scaler_path = os.path.join(data_dir, "scaler.npz")
  scaler = FeatureStandardizer.load(scaler_path)

  X_train, y_train = _load_split_arrays(data_dir, "train", scaler)
  X_val, y_val = _load_split_arrays(data_dir, "val", scaler)
  X_test, y_test = _load_split_arrays(data_dir, "test", scaler)

  # Light hand rotation on TRAIN only (using unscaled 63 coordinates -> angles -> scaler)
  if aug_copies > 0 and rot_z_deg > 0:
    X_train_raw = (
        X_train * scaler.std.astype(np.float32)
        + scaler.mean.astype(np.float32)
    )
    X63_train = X_train_raw[:, :NUM_COORDS]

    featurizer = JointAngleFeatures()
    augmenter = LandmarkAugmenter(
        rot_z_deg=rot_z_deg,
        rot_xy_deg=rot_xy_deg,
        jitter_std=0.0,
        seed=seed,
    )

    aug_x_list = [X_train]
    aug_y_list = [y_train]
    for _ in range(aug_copies):
      X_aug_std = augmenter.augment_features(X63_train, featurizer, scaler)
      aug_x_list.append(X_aug_std.astype(np.float32))
      aug_y_list.append(y_train)

    X_train = np.concatenate(aug_x_list, axis=0)
    y_train = np.concatenate(aug_y_list, axis=0)

  autotune = tf.data.AUTOTUNE

  train_ds = (
      tf.data.Dataset.from_tensor_slices((X_train, y_train))
      .shuffle(
          buffer_size=min(len(X_train), shuffle_buffer),
          seed=seed,
          reshuffle_each_iteration=True,
      )
      .batch(batch_size)
      .prefetch(autotune)
  )

  val_ds = (
      tf.data.Dataset.from_tensor_slices((X_val, y_val))
      .batch(batch_size)
      .prefetch(autotune)
  )

  test_ds = (
      tf.data.Dataset.from_tensor_slices((X_test, y_test))
      .batch(batch_size)
      .prefetch(autotune)
  )

  return train_ds, val_ds, test_ds
