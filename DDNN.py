import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model


# ============================================================
# 1. Basic Parameters
# ============================================================

SEED = 80                            # Random seed

DATASET_DIR = r"D:\Dataset"          # Training dataset directory
REAL_MOD_CSV = r"D:\modulation_spectrum.csv"       # Measured modulation spectrum path

LAMBDA_MIN = 400.0                   # Minimum wavelength (nm)
LAMBDA_MAX = 700.0                   # Maximum wavelength (nm)
LAMBDA_STEP = 1                      # Wavelength sampling interval (nm)

EPSILON = 1e-8                       # Avoid division by zero

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

wavelength = np.arange(
    LAMBDA_MIN,
    LAMBDA_MAX + LAMBDA_STEP / 2,
    LAMBDA_STEP,
    dtype=np.float32
)

N = len(wavelength)


# ============================================================
# 2. Network and Training Parameters
# ============================================================

BASE_FILTERS = 32                    # Number of convolutional feature channels
KERNEL_SIZE = 9                      # Convolution kernel size

DILATION_RATES = [
    1, 1, 2, 2, 4,
    4, 8, 8, 16, 1
]                                    # Dilation rates of the convolutional modules

DROPOUT = 0.0                        # Dropout rate
CLIP_NORM = 1.0                      # Gradient clipping threshold

EPOCHS = 800                         # Number of supervised training epochs
LEARNING_RATE = 1e-3                 # Adam learning rate
BATCH_SIZE = 16                      # Batch size
PRINT_EVERY = 100                    # Print training loss every 100 epochs


# ============================================================
# 3. Load Training Data
#
# All CSV files are stored directly in the same directory.
# The first six columns correspond to:
# Wavelength, S0, S1, S2, S3, and noisy modulation intensity.
# ============================================================

def load_training_data():

    files = sorted(Path(DATASET_DIR).glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files were found in the training directory."
        )

    X, Y = [], []

    for file in files:

        data = pd.read_csv(file).iloc[:, :6].to_numpy(
            dtype=np.float32
        )

        if data.shape[1] != 6 or not np.all(np.isfinite(data)):
            raise ValueError(
                f"Invalid data found in {file.name}."
            )

        if len(data) != N or not np.allclose(
            data[:, 0],
            wavelength,
            rtol=0,
            atol=1e-5
        ):
            raise ValueError(
                f"Wavelength grid mismatch in {file.name}."
            )

        # Use the noisy modulation spectrum as input
        X.append(data[:, 5])

        # Use the ground-truth Stokes spectra as supervised labels
        Y.append(data[:, 1:5])

    X = np.asarray(X, dtype=np.float32)[..., None]
    Y = np.asarray(Y, dtype=np.float32)

    # Use the maximum absolute S0 value in the training dataset
    # as the shared normalization scale for all Stokes components
    # and the modulation spectrum.
    scale = float(np.max(np.abs(Y[..., 0])))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Invalid normalization scale for the training S0 data."
        )

    print(f"Training samples: {len(X)}")

    return X / scale, Y / scale, scale


# ============================================================
# 4. Load the Measured Modulation Spectrum
#
# Column 1: Wavelength
# Column 2: Noisy modulation intensity
# ============================================================

def load_real_modulation():

    data = pd.read_csv(REAL_MOD_CSV).iloc[:, :2].to_numpy(
        dtype=np.float32
    )

    if data.shape[1] != 2 or not np.all(np.isfinite(data)):
        raise ValueError(
            "Invalid measured modulation spectrum data."
        )

    data = data[np.argsort(data[:, 0])]

    wl = data[:, 0]
    mod = data[:, 1]

    if len(wl) < 2 or len(np.unique(wl)) != len(wl):
        raise ValueError(
            "Invalid wavelength data in the measured modulation spectrum."
        )

    if wl.min() > LAMBDA_MIN or wl.max() < LAMBDA_MAX:
        raise ValueError(
            "The measured modulation spectrum does not cover "
            "the target wavelength range."
        )

    return np.asarray(mod, dtype=np.float32)


# ============================================================
# 5. All-CNN Network Architecture
# Consistent with the DA-PGNN architecture
# ============================================================

def stokes_activation(y):
    """Apply ReLU to S0 and tanh to S1, S2, and S3."""

    return tf.concat([
        tf.nn.relu(y[..., :1]),
        tf.nn.tanh(y[..., 1:])
    ], axis=-1)


def build_model():

    inputs = layers.Input(shape=(N, 1))

    # Initial convolutional layer
    x = layers.Conv1D(
        BASE_FILTERS,
        KERNEL_SIZE,
        padding="same",
        activation="relu"
    )(inputs)

    # Ten double-convolution modules
    for dilation in DILATION_RATES:

        x = layers.Conv1D(
            BASE_FILTERS,
            KERNEL_SIZE,
            padding="same",
            dilation_rate=dilation,
            activation="relu"
        )(x)

        if DROPOUT > 0:
            x = layers.Dropout(DROPOUT)(x)

        x = layers.Conv1D(
            BASE_FILTERS,
            KERNEL_SIZE,
            padding="same",
            activation="relu"
        )(x)

    # Final convolutional layer
    x = layers.Conv1D(
        BASE_FILTERS,
        KERNEL_SIZE,
        padding="same",
        activation="relu"
    )(x)

    # Four output channels corresponding to the Stokes parameters
    x = layers.Conv1D(
        4, 1, padding="same"
    )(x)

    outputs = layers.Lambda(stokes_activation)(x)

    return Model(inputs, outputs, name="DDNN_allCNN")


# ============================================================
# 6. Supervised Data Loss
#
# L_data = (1/K) * sum ||S_pred - S_true||^2
# ============================================================

@tf.function
def data_loss(y_true, y_pred):

    return tf.reduce_mean(
        tf.reduce_sum(
            tf.square(y_true - y_pred),
            axis=[1, 2]
        )
    )


# ============================================================
# 7. Print Training Loss Every 100 Epochs
# ============================================================

class TrainingMonitor(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        if (epoch + 1) % PRINT_EVERY == 0:

            print(
                f"Epoch {epoch + 1:4d}/{EPOCHS} | "
                f"Loss={logs.get('loss', np.nan):.8e}"
            )


# ============================================================
# 8. Train the DDNN
# ============================================================

def train_ddnn():

    start = time.perf_counter()

    X, Y, scale = load_training_data()

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=CLIP_NORM
        ),
        loss=data_loss
    )

    print("\n========== DDNN Training ==========")

    model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[TrainingMonitor()],
        verbose=0
    )

    print(
        f"Training time: {time.perf_counter() - start:.3f} s"
    )

    return model, scale


# ============================================================
# 9. Direct DDNN Prediction
#
# Input the measured modulation spectrum and directly predict
# the corresponding Stokes spectra.
# ============================================================

def predict_ddnn(model, scale):

    start = time.perf_counter()

    measured = load_real_modulation()

    # Apply the same normalization scale used during training
    x = (measured / scale)[None, :, None].astype(np.float32)

    # Direct network prediction
    prediction = model.predict(x, verbose=0)[0]

    # Restore the original intensity scale
    result = prediction * scale

    print(
        f"Prediction time: {time.perf_counter() - start:.4f} s"
    )

    return result


# ============================================================
# 10. Visualization
#
# Display a single 2 x 2 figure containing:
# S0, s1, s2, and s3.
# ============================================================

def plot_result(stokes):

    s0 = stokes[:, 0]

    # Calculate the three normalized Stokes components
    sn = stokes[:, 1:] / np.maximum(
        s0[:, None], EPSILON
    )

    curves = [
        s0,
        sn[:, 0],
        sn[:, 1],
        sn[:, 2]
    ]

    titles = [
        r"$S_0$",
        r"$s_1=S_1/S_0$",
        r"$s_2=S_2/S_0$",
        r"$s_3=S_3/S_0$"
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9)
    )

    for i, ax in enumerate(axes.flat):

        ax.plot(
            wavelength,
            curves[i],
            linewidth=1.8
        )

        ax.set_title(titles[i])
        ax.set_xlabel("Wavelength (nm)")

        ax.set_ylabel(
            "Intensity (a.u.)" if i == 0
            else "Normalized Stokes"
        )

        ax.set_xlim(LAMBDA_MIN, LAMBDA_MAX)
        ax.grid(alpha=0.3)

    fig.suptitle("DDNN Stokes Reconstruction")
    fig.tight_layout()
    plt.show()


# ============================================================
# 11. Main Program
# ============================================================

def main():

    start = time.perf_counter()

    # Train the network using the first-stage loss function
    model, scale = train_ddnn()

    # Reconstruct the measured modulation spectrum
    result = predict_ddnn(model, scale)

    print(
        f"\nTotal time: {time.perf_counter() - start:.3f} s"
    )

    plot_result(result)


if __name__ == "__main__":
    main()