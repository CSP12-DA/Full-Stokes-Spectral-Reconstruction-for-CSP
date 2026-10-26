
import time
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model


# ============================================================
# 1. Basic Parameters
# ============================================================

SEED = 80                            # Random seed

REAL_MOD_CSV = r"D:\modulation_spectrum.csv"  # Path to the measured modulation spectrum

LAMBDA_MIN = 400.0                   # Starting wavelength (nm)
LAMBDA_MAX = 700.0                   # Ending wavelength (nm)
LAMBDA_STEP = 1                    # Wavelength sampling interval (nm)

EPSILON = 1e-8                       # Prevent division by zero

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
# 2. Network and Optimization Parameters
# ============================================================

BASE_FILTERS = 32                    # Number of convolution filters
KERNEL_SIZE = 9                      # Kernel size

DILATION_RATES = [
    1, 1, 2, 2, 4,
    4, 8, 8, 16, 1
]                                     # Dilation rates

DROPOUT = 0.0                        # Dropout rate
CLIP_NORM = 1.0                      # Gradient clipping threshold

EPOCHS = 1000                         # Maximum number of physics-optimization iterations
MIN_EPOCHS = 100                     # Minimum number of optimization iterations
LEARNING_RATE = 1e-3                 # Adam learning rate
PATIENCE = 100                       # Early-stopping patience
MIN_DELTA = 1e-7                     # Minimum loss improvement
PRINT_EVERY = 100                    # Print losses every 100 iterations

ALPHA = 0.8                          # DoP constraint weight
BETA = 0.05                          # Second-order smoothness constraint weight


# ============================================================
# 3. PSIM Physical Model Parameters
# ============================================================

DELTA_N1 = 0.0090                    # Birefringence of the first retarder
DELTA_N2 = 0.0090                    # Birefringence of the second retarder

D1_MM = 4.0                          # Thickness of the first retarder (mm)
D2_MM = 8.0                          # Thickness of the second retarder (mm)

THETA1 = 0.0                         # Fast-axis angle of the first retarder (deg)
THETA2 = 45.0                        # Fast-axis angle of the second retarder (deg)
ANALYZER = 0.0                       # Analyzer transmission-axis angle (deg)


def retarder_mueller(delta, theta):
    """Mueller matrix of a linear retarder."""

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
    """Calculate the PSIM modulation coefficients from the system parameters."""

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

        # Phase retardations of the two retarders
        delta1 = 2 * np.pi * DELTA_N1 * D1_MM * 1e6 / lam
        delta2 = 2 * np.pi * DELTA_N2 * D2_MM * 1e6 / lam

        R1 = retarder_mueller(delta1, t1)
        R2 = retarder_mueller(delta2, t2)

        # First row of the system modulation matrix
        A[i] = analyzer_row @ R2 @ R1

    return tf.constant(
        A[None, :, :],
        dtype=tf.float32
    )


PSIM_A = build_psim_matrix()


def forward_psim(stokes):
    """Generate the predicted modulation spectrum using the PSIM forward model."""

    return tf.reduce_sum(
        PSIM_A * stokes,
        axis=-1,
        keepdims=True
    )


# ============================================================
# 4. Load the Noisy Modulation Spectrum
#
# Column 1: wavelength
# Column 2: noisy modulation intensity
#
# ============================================================

def load_real_modulation():

    data = pd.read_csv(REAL_MOD_CSV).iloc[:, :2].to_numpy(
        dtype=np.float32
    )

    if data.shape[1] != 2 or not np.all(np.isfinite(data)):
        raise ValueError("Invalid measured modulation spectrum data")

    data = data[np.argsort(data[:, 0])]

    wl = data[:, 0]
    mod = data[:, 1]

    if len(wl) < 2 or len(np.unique(wl)) != len(wl):
        raise ValueError("Invalid wavelength data in the measured modulation spectrum")

    if wl.min() > LAMBDA_MIN or wl.max() < LAMBDA_MAX:
        raise ValueError("The modulation spectrum does not cover the target wavelength range")

    return np.asarray(mod, dtype=np.float32)

# ============================================================
# 5. All-CNN Network
# Consistent with DDNN and DA-PGNN
# ============================================================

def stokes_activation(y):
    """Use ReLU for S0 and tanh for S1-S3."""

    return tf.concat([
        tf.nn.relu(y[..., :1]),
        tf.nn.tanh(y[..., 1:])
    ], axis=-1)


def build_model():

    inputs = layers.Input(shape=(N, 1))

    # Initial convolution
    x = layers.Conv1D(
        BASE_FILTERS,
        KERNEL_SIZE,
        padding="same",
        activation="relu"
    )(inputs)

    # 10 dual-convolution blocks
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

    # Final convolution
    x = layers.Conv1D(
        BASE_FILTERS,
        KERNEL_SIZE,
        padding="same",
        activation="relu"
    )(x)

    # Output the four Stokes components
    x = layers.Conv1D(
        4, 1, padding="same"
    )(x)

    outputs = layers.Lambda(stokes_activation)(x)

    return Model(inputs, outputs, name="PGNN_allCNN")


# ============================================================
# 6. Physics-Stage Loss Function
#
# L = L_phy + ALPHA*L_dop + BETA*L_smooth
#
# ============================================================

def normalized_stokes(stokes):
    """Normalized Stokes parameters: si = Si / S0."""

    s0 = tf.maximum(
        stokes[..., :1],
        EPSILON
    )

    return stokes[..., 1:] / s0


def physics_losses(model, measured, training=False):

    """
    L_phy    = ||I_pred-I_measured||²

    L_dop    = ||max(0,sqrt(s1²+s2²+s3²)-1)||²

    L_smooth = sum ||D2(si)||²
    """

    # Predict the Stokes parameters from the measured modulation spectrum
    stokes = model(
        measured,
        training=training
    )

    # Physics-consistency loss
    mod_pred = forward_psim(stokes)

    L_phy = tf.reduce_sum(
        tf.square(mod_pred - measured)
    )

    # Normalize the Stokes parameters
    sn = normalized_stokes(stokes)

    # DoP constraint
    dop = tf.sqrt(
        tf.reduce_sum(
            tf.square(sn),
            axis=-1
        ) + EPSILON
    )

    # Penalize the portion of DoP exceeding 1
    L_dop = tf.reduce_sum(
        tf.square(
            tf.nn.relu(dop - 1.0)
        )
    )

    # Apply a second-order smoothness constraint to the normalized Stokes parameters
    d2 = (
        sn[:, 2:, :]
        - 2.0 * sn[:, 1:-1, :]
        + sn[:, :-2, :]
    )

    L_smooth = tf.reduce_sum(
        tf.square(d2)
    )

    # Total physics loss
    total = (
        L_phy
        + ALPHA * L_dop
        + BETA * L_smooth
    )

    return total, L_phy, L_dop, L_smooth


# ============================================================
# 7. Physics-Driven PGNN Training
#
# Initialize the network with random weights.
# Perform physics-based optimization using the measured modulation spectrum.
#
# ============================================================

def train_pgnn():

    start = time.perf_counter()

    measured = load_real_modulation()

    # Normalize using the maximum absolute value of the input modulation spectrum
    scale = float(np.max(np.abs(measured)))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid modulation-spectrum normalization scale")

    x = (
        measured / scale
    )[None, :, None].astype(np.float32)

    x_tf = tf.constant(x, dtype=tf.float32)

    # Initialize the network randomly
    model = build_model()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM
    )

    best_loss = np.inf
    best_weights = None
    best_epoch = 0
    wait = 0

    print("\n========== PGNN Physics Training ==========")

    for epoch in range(1, EPOCHS + 1):

        with tf.GradientTape() as tape:

            total, _, _, _ = physics_losses(
                model,
                x_tf,
                training=True
            )

        # Compute gradients from the physics loss
        grads = tape.gradient(
            total,
            model.trainable_variables
        )

        optimizer.apply_gradients([
            (g, v)
            for g, v in zip(
                grads,
                model.trainable_variables
            )
            if g is not None
        ])

        # Recalculate the loss after updating the parameters
        total, phy, dop, smooth = physics_losses(
            model,
            x_tf,
            training=False
        )

        loss_value = float(total.numpy())

        if not np.isfinite(loss_value):
            raise FloatingPointError("The PGNN loss contains NaN or Inf")

        # Save the weights corresponding to the minimum loss
        if loss_value < best_loss - MIN_DELTA:

            best_loss = loss_value
            best_epoch = epoch
            best_weights = model.get_weights()
            wait = 0

        else:
            wait += 1

        # Print the results every 100 iterations
        if epoch % PRINT_EVERY == 0:

            print(
                f"Iter {epoch:4d}/{EPOCHS} | "
                f"Total={loss_value:.6e} | "
                f"Phy={float(phy.numpy()):.6e} | "
                f"DoP={float(dop.numpy()):.6e} | "
                f"Smooth={float(smooth.numpy()):.6e}"
            )

        # Check the early-stopping condition
        if (
            epoch >= MIN_EPOCHS
            and wait >= PATIENCE
        ):

            print(f"Early stopped at iteration {epoch}")
            break

    # Restore the best model weights
    if best_weights is not None:
        model.set_weights(best_weights)

    # Network prediction
    pred_start = time.perf_counter()

    prediction = model.predict(
        x,
        verbose=0
    )[0]

    prediction_time = time.perf_counter() - pred_start

    # Restore the original intensity scale of the modulation spectrum
    result = prediction * scale

    print(f"\nBest iteration: {best_epoch}")
    print(f"Best loss: {best_loss:.8e}")
    print(f"Prediction time: {prediction_time:.4f}s")
    print(
        f"Optimization time: {time.perf_counter()-start:.3f}s"
    )

    return result


# ============================================================
# 8. Result Visualization
#
# One 2x2 figure:
# S0、s1、s2、s3
#
# ============================================================

def plot_result(stokes):

    s0 = stokes[:, 0]

    # Calculate the three normalized Stokes parameters
    sn = stokes[:, 1:] / np.maximum(
        s0[:, None],
        EPSILON
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
        2, 2,
        figsize=(13, 9)
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

        ax.set_xlim(
            LAMBDA_MIN,
            LAMBDA_MAX
        )

        ax.grid(alpha=0.3)

    fig.suptitle("PGNN Stokes Reconstruction")

    fig.tight_layout()
    plt.show()


# ============================================================
# 9. Main Program
# ============================================================

def main():

    start = time.perf_counter()

    # Initialize randomly and perform physics-constrained optimization
    result = train_pgnn()

    print(f"\nTotal time: {time.perf_counter()-start:.3f}s")

    plot_result(result)


if __name__ == "__main__":
    main()
