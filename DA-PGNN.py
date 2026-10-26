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
REAL_MOD_CSV = r"D:\modulation_spectrum.csv"  # Real modulation spectrum file

LAMBDA_MIN = 400.0                   # Starting wavelength (nm)
LAMBDA_MAX = 700.0                   # Ending wavelength (nm)
LAMBDA_STEP = 1                      # Wavelength sampling interval (nm)
EPSILON = 1e-8                       # Numerical stability constant

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

BASE_FILTERS = 32                    # Number of convolutional filters
KERNEL_SIZE = 9                      # Convolution kernel size
DILATION_RATES = [1, 1, 2, 2, 4, 4, 8, 8, 16, 1]  # Dilation rates
DROPOUT = 0.0                        # Dropout rate
CLIP_NORM = 1.0                      # Gradient clipping threshold

PRETRAIN_EPOCHS = 800                # Number of pretraining epochs
PRETRAIN_LR = 1e-3                   # Pretraining learning rate
BATCH_SIZE = 16                      # Batch size

FINETUNE_EPOCHS = 1000               # Maximum number of fine-tuning iterations
FINETUNE_MIN_EPOCHS = 100            # Minimum number of fine-tuning iterations
FINETUNE_LR = 1e-3                   # Fine-tuning learning rate
FINETUNE_PATIENCE = 100              # Early stopping patience
FINETUNE_MIN_DELTA = 1e-7            # Minimum loss improvement

PRINT_EVERY = 100                    # Loss logging interval for both stages

ALPHA = 0.8                          # DoP constraint weight
BETA = 0.05                          # Smoothness constraint weight


# ============================================================
# 3. PSIM Physical Model Parameters
# ============================================================

DELTA_N1 = 0.0090                    # Birefringence of the first retarder
DELTA_N2 = 0.0090                    # Birefringence of the second retarder

D1_MM = 4.0                          # Thickness of the first retarder (mm)
D2_MM = 8.0                          # Thickness of the second retarder (mm)

THETA1 = 0.0                         # Fast-axis angle of the first retarder (degrees)
THETA2 = 45.0                        # Fast-axis angle of the second retarder (degrees)
ANALYZER = 0.0                       # Analyzer transmission-axis angle (degrees)


def retarder_mueller(delta, theta):
    """Return the Mueller matrix of a linear retarder."""

    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    cd = np.cos(delta)
    sd = np.sin(delta)

    return np.array([
        [1, 0, 0, 0],
        [0, c*c + s*s*cd, c*s*(1-cd), -s*sd],
        [0, c*s*(1-cd), s*s + c*c*cd, c*sd],
        [0, s*sd, -c*sd, cd]
    ], dtype=np.float32)


def build_psim_matrix():
    """Calculate the PSIM modulation coefficients at each wavelength."""

    t1 = np.deg2rad(THETA1)
    t2 = np.deg2rad(THETA2)
    a = np.deg2rad(ANALYZER)

    ca = np.cos(2 * a)
    sa = np.sin(2 * a)

    # First row of the analyzer Mueller matrix
    analyzer_row = 0.5 * np.array(
        [1, ca, sa, 0],
        dtype=np.float32
    )

    A = np.zeros((N, 4), dtype=np.float32)

    for i, lam in enumerate(wavelength):

        delta1 = 2 * np.pi * DELTA_N1 * D1_MM * 1e6 / lam
        delta2 = 2 * np.pi * DELTA_N2 * D2_MM * 1e6 / lam

        R1 = retarder_mueller(delta1, t1)
        R2 = retarder_mueller(delta2, t2)

        A[i] = analyzer_row @ R2 @ R1

    return tf.constant(A[None, :, :], dtype=tf.float32)


PSIM_A = build_psim_matrix()


def forward_psim(stokes):
    """Calculate the modulation spectrum from the predicted Stokes parameters."""

    return tf.reduce_sum(
        PSIM_A * stokes,
        axis=-1,
        keepdims=True
    )


# ============================================================
# 4. Data Loading
# ============================================================

def load_training_data():
    """
    Load CSV files stored directly in the dataset directory.

    The first six columns contain:
    wavelength, S0, S1, S2, S3, and noisy modulation spectrum.
    """

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
                f"Invalid data found in file: {file.name}"
            )

        if len(data) != N or not np.allclose(
            data[:, 0], wavelength, rtol=0, atol=1e-5
        ):
            raise ValueError(
                f"Wavelength grid mismatch in file: {file.name}"
            )

        Y.append(data[:, 1:5])
        X.append(data[:, 5])

    X = np.asarray(X, dtype=np.float32)[..., None]
    Y = np.asarray(Y, dtype=np.float32)

    # Use the maximum absolute S0 value in the training set
    # as the common normalization scale
    scale = float(np.max(np.abs(Y[..., 0])))

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(
            "Invalid S0 normalization scale in the training dataset."
        )

    print(f"Number of training samples: {len(X)}")

    return X / scale, Y / scale, scale


def load_real_modulation():
    """
    Load the real noisy modulation spectrum.

    The first column contains wavelength, and the second column
    contains modulation intensity.
    """

    data = pd.read_csv(REAL_MOD_CSV).iloc[:, :2].to_numpy(
        dtype=np.float32
    )

    if data.shape[1] != 2 or not np.all(np.isfinite(data)):
        raise ValueError("Invalid real modulation spectrum data.")

    data = data[np.argsort(data[:, 0])]
    wl, mod = data[:, 0], data[:, 1]

    if len(wl) < 2 or len(np.unique(wl)) != len(wl):
        raise ValueError("Invalid wavelength data in the real modulation spectrum.")

    if wl.min() > LAMBDA_MIN or wl.max() < LAMBDA_MAX:
        raise ValueError(
            "The real modulation spectrum does not cover the target wavelength range."
        )

    return np.asarray(mod, dtype=np.float32)


# ============================================================
# 5. All-CNN Network
# ============================================================

def stokes_activation(y):
    """Apply ReLU to S0 and tanh to S1, S2, and S3."""

    return tf.concat([
        tf.nn.relu(y[..., :1]),
        tf.nn.tanh(y[..., 1:])
    ], axis=-1)


def build_model():
    """Build a 1D CNN without pooling or fully connected layers."""

    inputs = layers.Input(shape=(N, 1))

    x = layers.Conv1D(
        BASE_FILTERS, KERNEL_SIZE,
        padding="same", activation="relu"
    )(inputs)

    for dilation in DILATION_RATES:

        x = layers.Conv1D(
            BASE_FILTERS, KERNEL_SIZE,
            padding="same",
            dilation_rate=dilation,
            activation="relu"
        )(x)

        if DROPOUT > 0:
            x = layers.Dropout(DROPOUT)(x)

        x = layers.Conv1D(
            BASE_FILTERS, KERNEL_SIZE,
            padding="same", activation="relu"
        )(x)

    x = layers.Conv1D(
        BASE_FILTERS, KERNEL_SIZE,
        padding="same", activation="relu"
    )(x)

    x = layers.Conv1D(
        4, 1, padding="same"
    )(x)

    outputs = layers.Lambda(stokes_activation)(x)

    return Model(inputs, outputs, name="DA_PGNN_allCNN")


# ============================================================
# 6. Stage 1: Data-Supervised Loss
# ============================================================

@tf.function
def data_loss(y_true, y_pred):
    """
    Data-supervised loss:

    L_data = (1/K) * sum ||S_pred - S_true||^2
    """

    return tf.reduce_mean(
        tf.reduce_sum(
            tf.square(y_true - y_pred),
            axis=[1, 2]
        )
    )


# ============================================================
# 7. Stage 2: Physics-Based Loss
# ============================================================

def normalized_stokes(stokes):
    """Calculate normalized Stokes parameters: si = Si / S0."""

    s0 = tf.maximum(stokes[..., :1], EPSILON)
    return stokes[..., 1:] / s0


def physics_losses(model, measured, training=False):
    """
    Fine-tuning loss:

    L_fine = L_phy + ALPHA * L_dop + BETA * L_smooth

    L_phy    = ||I_pred - I_measured||^2
    L_dop    = ||max(0, sqrt(s1^2 + s2^2 + s3^2) - 1)||^2
    L_smooth = sum ||D2(si)||^2
    """

    stokes = model(measured, training=training)

    # Physical consistency loss
    mod_pred = forward_psim(stokes)
    L_phy = tf.reduce_sum(
        tf.square(mod_pred - measured)
    )

    # Normalized Stokes parameters
    sn = normalized_stokes(stokes)

    # Degree of polarization (DoP) constraint
    dop = tf.sqrt(
        tf.reduce_sum(tf.square(sn), axis=-1) + EPSILON
    )

    L_dop = tf.reduce_sum(
        tf.square(tf.nn.relu(dop - 1.0))
    )

    # Second-order smoothness constraint
    d2 = sn[:, 2:, :] - 2 * sn[:, 1:-1, :] + sn[:, :-2, :]

    L_smooth = tf.reduce_sum(tf.square(d2))

    total = L_phy + ALPHA * L_dop + BETA * L_smooth

    return total, L_phy, L_dop, L_smooth


# ============================================================
# 8. Stage 1: Data-Driven Pretraining
# ============================================================

class TrainingMonitor(tf.keras.callbacks.Callback):
    """Print the pretraining loss every 100 epochs."""

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        if (epoch + 1) % PRINT_EVERY == 0:

            print(
                f"Epoch {epoch+1:4d}/{PRETRAIN_EPOCHS} | "
                f"Loss={logs.get('loss', np.nan):.8e}"
            )


def pretrain():
    """Perform supervised pretraining using all training samples."""

    start = time.perf_counter()

    X, Y, scale = load_training_data()

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=PRETRAIN_LR,
            clipnorm=CLIP_NORM
        ),
        loss=data_loss
    )

    print("\n========== Pretraining ==========")

    model.fit(
        X, Y,
        epochs=PRETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[TrainingMonitor()],
        verbose=0
    )

    print(
        f"Pretraining time: {time.perf_counter()-start:.3f} s"
    )

    return model, scale


# ============================================================
# 9. Stage 2: Physics-Adaptive Fine-Tuning
# ============================================================

def finetune(model, scale):
    """Perform physics-adaptive fine-tuning using the real modulation spectrum."""

    start = time.perf_counter()

    measured = load_real_modulation()

    # Use the same intensity normalization scale as in pretraining
    x = (measured / scale)[None, :, None].astype(np.float32)
    x_tf = tf.constant(x)

    # Initialize the fine-tuning network with pretrained weights
    fine_model = build_model()
    fine_model.set_weights(model.get_weights())

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=FINETUNE_LR,
        clipnorm=CLIP_NORM
    )

    best_loss = np.inf
    best_weights = None
    best_epoch = 0
    wait = 0

    print("\n========== Physics Fine-Tuning ==========")

    for epoch in range(1, FINETUNE_EPOCHS + 1):

        with tf.GradientTape() as tape:

            total, _, _, _ = physics_losses(
                fine_model, x_tf, training=True
            )

        grads = tape.gradient(
            total, fine_model.trainable_variables
        )

        optimizer.apply_gradients([
            (g, v)
            for g, v in zip(grads, fine_model.trainable_variables)
            if g is not None
        ])

        # Recalculate the loss after the parameter update
        total, phy, dop, smooth = physics_losses(
            fine_model, x_tf, training=False
        )

        loss_value = float(total.numpy())

        if not np.isfinite(loss_value):
            raise FloatingPointError(
                "NaN or Inf detected in the fine-tuning loss."
            )

        if loss_value < best_loss - FINETUNE_MIN_DELTA:

            best_loss = loss_value
            best_epoch = epoch
            best_weights = fine_model.get_weights()
            wait = 0

        else:
            wait += 1

        # Print the losses every 100 iterations
        if epoch % PRINT_EVERY == 0:

            print(
                f"Iteration {epoch:4d}/{FINETUNE_EPOCHS} | "
                f"Total={loss_value:.6e} | "
                f"Physics={float(phy.numpy()):.6e} | "
                f"DoP={float(dop.numpy()):.6e} | "
                f"Smoothness={float(smooth.numpy()):.6e}"
            )

        if (
            epoch >= FINETUNE_MIN_EPOCHS
            and wait >= FINETUNE_PATIENCE
        ):
            print(f"Early stopping triggered at iteration {epoch}.")
            break

    # Restore the best model weights
    if best_weights is not None:
        fine_model.set_weights(best_weights)

    # Restore the original intensity scale
    result = fine_model(x_tf, training=False).numpy()[0] * scale

    print(f"Best iteration: {best_epoch}")
    print(f"Best loss: {best_loss:.8e}")
    print(
        f"Fine-tuning time: {time.perf_counter()-start:.3f} s"
    )

    return result


# ============================================================
# 10. Result Visualization
# ============================================================

def plot_result(stokes):
    """Plot S0 and the three normalized Stokes parameters."""

    s0 = stokes[:, 0]

    # Retain only numerical protection against division by zero
    sn = stokes[:, 1:] / np.maximum(s0[:, None], EPSILON)

    curves = [s0, sn[:, 0], sn[:, 1], sn[:, 2]]

    titles = [
        r"$S_0$",
        r"$s_1=S_1/S_0$",
        r"$s_2=S_2/S_0$",
        r"$s_3=S_3/S_0$"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i, ax in enumerate(axes.flat):

        ax.plot(wavelength, curves[i], linewidth=1.8)
        ax.set_title(titles[i])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(
            "Intensity (a.u.)" if i == 0 else "Normalized Stokes"
        )
        ax.set_xlim(LAMBDA_MIN, LAMBDA_MAX)
        ax.grid(alpha=0.3)

    fig.suptitle("DA-PGNN Stokes Reconstruction")
    fig.tight_layout()
    plt.show()


# ============================================================
# 11. Main Program
# ============================================================

def main():

    start = time.perf_counter()

    model, scale = pretrain()

    result = finetune(model, scale)

    print(f"\nTotal execution time: {time.perf_counter()-start:.3f} s")

    plot_result(result)


if __name__ == "__main__":
    main()