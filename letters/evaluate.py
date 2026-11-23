import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from letters.data import build_datasets


def evaluate_dataset(model, ds, split_name="val"):
  y_true_list = []
  y_pred_list = []

  for x_batch, y_batch in ds:
    preds = model.predict(x_batch, verbose=0)
    pred_ids = np.argmax(preds, axis=-1)
    y_true_list.append(y_batch.numpy())
    y_pred_list.append(pred_ids)

  y_true = np.concatenate(y_true_list, axis=0)
  y_pred = np.concatenate(y_pred_list, axis=0)

  acc = accuracy_score(y_true, y_pred)
  precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
      y_true, y_pred, average="macro", zero_division=0
  )

  print(f"\n=== Evaluation Results ({split_name}) ===")
  print(f"Accuracy       : {acc:.4f}")
  print(f"Precision Macro: {precision_macro:.4f}")
  print(f"Recall Macro   : {recall_macro:.4f}")
  print(f"F1 Macro       : {f1_macro:.4f}")

  print("\n--- Per-Class Classification Report ---")
  print(classification_report(y_true, y_pred, zero_division=0))

  cm = confusion_matrix(y_true, y_pred)
  print("--- Confusion Matrix ---")
  print(cm)

  return {
      "accuracy": acc,
      "precision_macro": precision_macro,
      "recall_macro": recall_macro,
      "f1_macro": f1_macro,
      "confusion_matrix": cm,
  }


def main():
  parser = argparse.ArgumentParser(description="Evaluate letters model.")
  parser.add_argument("--model-path", required=True, help="Path to .keras model")
  parser.add_argument(
      "--data-dir", required=True, help="Path to processed_data/letters/final"
  )
  parser.add_argument(
      "--split",
      default="val",
      choices=["val", "test"],
      help="Dataset split to evaluate",
  )
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  train_ds, val_ds, test_ds = build_datasets(
      data_dir=args.data_dir,
      batch_size=args.batch_size,
      seed=args.seed,
      aug_copies=0,
  )

  model = tf.keras.models.load_model(args.model_path)

  target_ds = val_ds if args.split == "val" else test_ds
  evaluate_dataset(model, target_ds, split_name=args.split)


if __name__ == "__main__":
  main()
