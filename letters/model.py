"""Residual MLP classifier for static Arabic sign-language letters.

Input : 89 standardized features (63 landmark coordinates + 26 joint angles).
Output: softmax probabilities over 32 letter classes.
"""
from tensorflow import keras
from tensorflow.keras import layers

NUM_FEATURES = 89
NUM_CLASSES = 32


def residual_block(x, width: int, dropout: float, name: str):
    """Pre-activation residual block: (BN -> GELU -> Dense -> Dropout) x2 + skip connection."""
    shortcut = x

    h = layers.BatchNormalization(name=f"{name}_bn1")(x)
    h = layers.Activation("gelu", name=f"{name}_act1")(h)
    h = layers.Dense(width, name=f"{name}_dense1")(h)
    h = layers.Dropout(dropout, name=f"{name}_drop1")(h)

    h = layers.BatchNormalization(name=f"{name}_bn2")(h)
    h = layers.Activation("gelu", name=f"{name}_act2")(h)
    h = layers.Dense(width, name=f"{name}_dense2")(h)
    h = layers.Dropout(dropout, name=f"{name}_drop2")(h)

    # Skip connection: both tensors have the same width, so a plain add works
    return layers.Add(name=f"{name}_add")([shortcut, h])


def build_model(
    num_features: int = NUM_FEATURES,
    num_classes: int = NUM_CLASSES,
    width: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.2,
) -> keras.Model:
    """Build the residual MLP. width / num_blocks / dropout are the tunable knobs."""
    inputs = keras.Input(shape=(num_features,), name="features")

    # Project the 89 features to a fixed width so the residual adds line up
    x = layers.Dense(width, name="input_dense")(inputs)
    x = layers.BatchNormalization(name="input_bn")(x)
    x = layers.Activation("gelu", name="input_act")(x)

    for i in range(num_blocks):
        x = residual_block(x, width, dropout, name=f"block{i + 1}")

    # Classification head
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("gelu", name="head_act")(x)
    x = layers.Dropout(dropout, name="head_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return keras.Model(inputs, outputs, name="letters_resmlp")
