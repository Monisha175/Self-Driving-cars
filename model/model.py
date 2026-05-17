"""
model.py — PilotNet-inspired CNN for end-to-end autonomous steering.

Architecture based on:
  "End to End Learning for Self-Driving Cars" — Bojarski et al., NVIDIA (2016)
  https://arxiv.org/abs/1604.07316

Extended with:
  - Batch Normalization after each conv layer
  - ELU activations (smoother gradients for regression vs ReLU)
  - Dropout regularisation
  - Keras Lambda layers for normalisation and cropping (run on GPU)
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


def build_model(input_shape: tuple = (160, 320, 3)) -> Model:
    """
    Build and return the PilotNet CNN model.

    Input shape: (height, width, channels) — expects raw BGR/YUV simulator frames.
    Output: single float — predicted steering angle in range [-1, 1].
            Multiply by max_steering_angle (e.g. 25°) to get degrees.

    Args:
        input_shape: HxWxC of the input image. Default matches Udacity simulator output.

    Returns:
        Compiled Keras Model.
    """
    model = keras.Sequential([
        # ── Preprocessing (runs on GPU, faster than CPU preprocessing) ────────
        # Normalise pixel values from [0, 255] to [-1, 1]
        layers.Lambda(lambda x: (x / 127.5) - 1.0,
                      input_shape=input_shape,
                      name="normalise"),

        # Crop out the sky (top 70px) and car hood (bottom 25px)
        # Keeps only the road-relevant region of the frame
        layers.Cropping2D(cropping=((70, 25), (0, 0)), name="crop"),

        # ── Convolutional feature extractor ───────────────────────────────────
        # Block 1–3: strided convolutions for spatial downsampling
        layers.Conv2D(24, (5, 5), strides=(2, 2), padding="valid", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.ELU(name="elu1"),

        layers.Conv2D(36, (5, 5), strides=(2, 2), padding="valid", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.ELU(name="elu2"),

        layers.Conv2D(48, (5, 5), strides=(2, 2), padding="valid", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.ELU(name="elu3"),

        # Block 4–5: non-strided convolutions for fine feature extraction
        layers.Conv2D(64, (3, 3), padding="valid", name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.ELU(name="elu4"),

        layers.Conv2D(64, (3, 3), padding="valid", name="conv5"),
        layers.BatchNormalization(name="bn5"),
        layers.ELU(name="elu5"),

        layers.Dropout(0.2, name="drop_conv"),   # light dropout after conv stack

        # ── Fully connected decision layers ───────────────────────────────────
        layers.Flatten(name="flatten"),

        layers.Dense(1164, name="fc1"),
        layers.ELU(name="elu_fc1"),
        layers.Dropout(0.5, name="drop_fc1"),    # heaviest dropout here

        layers.Dense(100, name="fc2"),
        layers.ELU(name="elu_fc2"),
        layers.Dropout(0.3, name="drop_fc2"),

        layers.Dense(50, name="fc3"),
        layers.ELU(name="elu_fc3"),

        layers.Dense(10, name="fc4"),
        layers.ELU(name="elu_fc4"),

        # ── Output: steering angle ────────────────────────────────────────────
        # Linear activation — this is a regression task, not classification
        layers.Dense(1, activation="linear", name="steering_output"),
    ], name="pilotnet_cnn")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",          # Mean Squared Error standard for steering regression
        metrics=["mae"],     # Mean Absolute Error is more interpretable
    )

    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"\nTotal parameters:     {m.count_params():>12,}")
    print(f"Trainable parameters: {sum(w.numpy().size for w in m.trainable_weights):>12,}")
