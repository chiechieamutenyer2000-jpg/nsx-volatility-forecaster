"""Optional LSTM forecaster (needs TensorFlow). The rest of the system works without it."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on environment
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    AVAILABLE = True
except Exception:  # pragma: no cover
    tf = None
    AVAILABLE = False

WINDOW = 60


def _windows(r2: np.ndarray, w: int):
    X = np.stack([r2[i - w:i] for i in range(w, len(r2))])
    return X[..., None], r2[w:]


def lstm_forecast(y, n_test: int, window: int = WINDOW, epochs: int = 30, seed: int = 11, progress=None):
    """Train once on all data before the test period; forecast each test day's variance.

    The scaler is fitted on the training data only, so no future information leaks in.
    Returns an array of length n_test.
    """
    if not AVAILABLE:
        raise RuntimeError("TensorFlow is not installed, so the LSTM is unavailable. "
                           "Install it with: pip install tensorflow")
    y = np.asarray(y, float)
    r2 = y ** 2
    first = len(y) - n_test
    if first < window + 200:
        raise ValueError("Not enough training data for the LSTM.")
    clip = np.percentile(r2[:first], 99.5)
    r2c = np.minimum(r2, clip)
    scale = r2c[:first].std() or 1.0
    z = r2c / scale
    X, t = _windows(z[:first], window)
    nval = int(0.2 * len(X))
    Xtr, ytr, Xva, yva = X[:-nval], t[:-nval], X[-nval:], t[-nval:]
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((window, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="softplus")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0, callbacks=cb)
    if progress:
        progress(0.9)
    Xt = np.stack([z[i - window:i] for i in range(first, len(y))])[..., None]
    pred = m.predict(Xt, verbose=0).ravel() * scale
    return np.maximum(pred, 1e-6)
