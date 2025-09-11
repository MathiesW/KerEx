import tensorflow as tf
from typing import Tuple
from functools import partial
from math import pi


def _get_complex_tensor_from_tuple(x):
    """
    Get a complex tensor from a tuple of two real-valued tensors.

    Parameters
    ----------
    x : tf.Tensor | list | tuple
        Input tensor, either `tf.Tensor` (real part) or tuple/list (real- and imaginary part).
        If only the real part is provided, the imaginary part is assumed 0.

    Returns
    -------
    y : tf.Tensor
        Tensor with `dtype=tf.complex`.

    Raises
    ------
    ValueError
        If shapes of real- and imaginary part to not match.

    """

    if not isinstance(x, (tuple, list)) or len(x) != 2:
        real = x
        imag = tf.zeros_like(x)
    else:
        real, imag = x
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )

    complex_input = tf.complex(real=real, imag=imag)
    return complex_input


# === FFT ===
def _fft(x: tuple, fn: callable) -> Tuple[tf.Tensor, tf.Tensor]:
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = fn(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


# === real valued FFT ===
def _rfft(x: tf.Tensor, fn: callable) -> Tuple[tf.Tensor, tf.Tensor]:
    complex_output = fn(x)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def _irfft(x: tf.Tensor, fn: callable, n: tuple = None) -> tf.Tensor:
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = fn(complex_input, fft_length=n)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


# === FFT derivatives ===
def fftfreq(n, d, rad=False):
    fs = 1.0 / d
    df = fs / tf.cast(n, tf.float32)
    fft_freqs = tf.cast(tf.range(n) - n // 2, tf.float32)
    fft_freqs *= df

    if rad:
        fft_freqs *= (2.0 * pi)

    fft_freqs = tf.roll(fft_freqs, shift=n // 2, axis=0)
    return fft_freqs


# === FFT derivative ===
def fft_derivative(x, axis=-1, d=1.0, n=1):
    """
    Derivative in Fourier space

    Parameters
    ----------
    x : tf.Tensor
        Input array, can be either real or complex valued.
    axis : int (optional)
        Axis along which `x` is derived.
        Defaults to `-1`.
    d : float (optional)
        Discretization of `x` along the dimension of `axis`.
        Defaults to `1.0`.
    n : int (optional)
        Order of differentiation.
        Defaults to `1`.

    Returns
    -------
    dx : tf.Tensor
        nth derivative of `x` along `axis`.

    Examples
    --------
    >>> n = 32
    >>> t = tf.range(0, 2 * pi, 2 * pi / 32)
    >>> dt = 2 * pi / 32
    >>> x = tf.sin(t)
    >>> dxdt_ref = tf.cos(t)
    >>> dxdt = fft_derivative(x, d=dt)
    >>> tf.reduce_max(dxdt_ref - dxdt)  
    tf.Tensor(1.0728836e-06, shape=(), dtype=float32)

    Notes
    -----
    For higher orders, the derivatives may get noisy.
    The intensity depends on the signal length, i.e.,
    this effect is less significant for short signals.

    """
    
    nx = tf.shape(x)[axis]
    dim = len(tf.shape(x))

    # tf.signal.fft is always along last axis --> move axis to last dimension
    if dim > 2:
        perm = list(range(dim))
        perm.append(perm.pop(axis))

        x = tf.transpose(x, perm=perm)

    x = _get_complex_tensor_from_tuple(x)
    freqs = fftfreq(n=nx, d=d, rad=True)

    complex_diff = tf.signal.ifft(tf.pow((1j * tf.cast(freqs, x.dtype)), tf.cast(n, x.dtype)) * tf.signal.fft(x))
    diff = tf.math.real(complex_diff)

    # revert permutation
    if dim > 2:
        perm = list(range(dim))
        perm.insert(axis + 1, perm.pop())
        diff = tf.transpose(diff, perm=perm)

    return diff


# === derived functions
def fft_fn(x):
    return partial(_fft, fn=tf.signal.fft)(x)


def fft2_fn(x):
    return partial(_fft, fn=tf.signal.fft2d)(x)


def fft3_fn(x):
    return partial(_fft, fn=tf.signal.fft3d)(x)


def ifft_fn(x):
    return partial(_fft, fn=tf.signal.ifft)(x)


def ifft2_fn(x):
    return partial(_fft, fn=tf.signal.ifft2d)(x)


def ifft3_fn(x):
    return partial(_fft, fn=tf.signal.ifft3d)(x)


def rfft_fn(x):
    return partial(_rfft, fn=tf.signal.rfft)(x)


def rfft2_fn(x):
    return partial(_rfft, fn=tf.signal.rfft2d)(x)


def rfft3_fn(x):
    return partial(_rfft, fn=tf.signal.rfft3d)(x)


def irfft_fn(x, n=None):
    y_real, _ = partial(_irfft, fn=tf.signal.irfft, n=n)(x)
    return y_real


def irfft2_fn(x, n=None):
    y_real, _ = partial(_irfft, fn=tf.signal.irfft2d, n=n)(x)
    return y_real


def irfft3_fn(x, n=None):
    y_real, _ = partial(_irfft, fn=tf.signal.irfft3d, n=n)(x)
    return y_real
