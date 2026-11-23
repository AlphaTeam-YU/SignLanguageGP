"""Training script for the letters residual MLP.

Uses the TRAIN and VALIDATION splits only. The test split returned by
build_datasets() is deliberately ignored here and never evaluated.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from letters.data import build_datasets
from letters.model import NUM_CLASSES, build_model


class LogEvery(keras.callbacks.Callback):
    """Print a one-line summary every N epochs (keeps the notebook output short)."""

    def __init__(self, every: int = 5):
        super().__init__()
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.every == 0:
            print(f"epoch {epoch + 1:3d} | loss {logs['loss']:.4f} | acc {logs['accuracy']:.4f}"
                  f" | val_loss {logs['val_loss']:.4f} | val_acc {logs['val_accuracy']:.4f}")


def train(data_dir, out_dir, tag="baseline", width=256, num_blocks=3, dropout=0.2,
          epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
          label_smoothing=0.1, patience=8, seed=42):
    keras.utils.set_random_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # test_ds is ignored on purpose (`_`): the test set is used once, at the very end, by C
    train_ds, val_ds, _ = build_datasets(data_dir, batch_size=batch_size, seed=seed)

    # Integer labels -> one-hot, needed for CategoricalCrossentropy with label smoothing
    to_onehot = lambda x, y: (x, tf.one_hot(y, NUM_CLASSES))
    train_ds = train_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    # Cosine decay over the whole planned training run
    steps_per_epoch = int(train_ds.cardinality().numpy())
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=epochs * steps_per_epoch, alpha=0.01)

    model = build_model(width=width, num_blocks=num_blocks, dropout=dropout)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        # Stop when val accuracy stops improving and go back to the best weights
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=patience, restore_best_weights=True),
        LogEvery(5),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=callbacks, verbose=0)

    val_acc = hist.history["val_accuracy"]
    result = {
        "tag": tag, "width": width, "num_blocks": num_blocks, "dropout": dropout,
        "lr": lr, "weight_decay": weight_decay, "label_smoothing": label_smoothing,
        "epochs_run": len(val_acc),
        "best_epoch": int(np.argmax(val_acc)) + 1,
        "best_val_acc": float(np.max(val_acc)),
        "train_acc_at_best": float(hist.history["accuracy"][int(np.argmax(val_acc))]),
    }

    # Weights are already restored to the best epoch, so this saves the best model
    model.save(os.path.join(out_dir, f"{tag}.keras"))
    with open(os.path.join(out_dir, f"{tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Save per-epoch history so it can be plotted later
    with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)
    return model, result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    _, res = train(a.data_dir, a.out_dir, a.tag, a.width, a.num_blocks, a.dropout, a.epochs)
    print(res)
