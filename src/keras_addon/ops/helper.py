import inspect
from keras import ops
from keras import layers
from keras import saving
from keras import KerasTensor
from keras.src import backend
import string
from math import pi
import re


def fftfreq(n, d=1, rad=False):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to `1`.
    rad : bool, optional
        If this is set, the angular frequency `omega=2*pi*f` is returned.
        Defaults to `False`.

    Returns
    -------
    f : KerasTensor
        Tensor of length `n` containing the sample frequencies.

    Examples
    --------
    >>> from keras import ops
    >>> from ssp.keras.ops import fft, fftfreq
    >>> signal = ops.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = fft(signal)
    >>> n = ops.size(signal)
    >>> timestep = 0.1
    >>> freq = fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """

    fs = 1.0 / d
    df = fs / ops.cast(n, float)
    fft_freqs = ops.arange(-ops.cast(n // 2, float) * df, ops.cast(n // 2, float) * df, df)

    if rad:
        fft_freqs *= (2 * pi)

    return ops.roll(fft_freqs, shift=n // 2)


def squeeze_or_expand_to_same_rank(x1, x2, axis=-1, expand_rank_1: bool = True) -> tuple:
    """
    Squeeze/expand along `axis` if ranks differ from expected by exactly 1.

    Parameters
    ----------
    x1 : KerasTensor
        first input tensor
    x2 : KerasTensor
        second input tensor
    axis : int, optional
        axis to squeeze or expand along. Defaults to `-1`.
    expand_rank_1: bool, optional
        Defaults to `True`

    Returns
    -------
    x1, x2 : (KerasTensor, KerasTensor)
        Tuple of `(x1, x2)` with the same shape

    """

    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[axis] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = ops.expand_dims(x2, axis=axis)
            else:
                x1 = ops.squeeze(x1, axis=axis)
    if x2_rank == x1_rank + 1:
        if x2.shape[axis] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = ops.expand_dims(x1, axis=axis)
            else:
                x2 = ops.squeeze(x2, axis=axis)
    return x1, x2


def large_negative_number(dtype):
    """
    Return a Large negative number based on dtype.

    Parameters
    ----------
    dtype : str
        dtype of large negative number to return
    
    Returns
    -------
    c : float
        Large negative number with dtype `dtype` (-1e9 for `dtype="float32"`, -3e4 for `dtype=float16"`).
    
    """
    if backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


def index_to_einsum_variable(i):
    """Coverts an index to a einsum variable name.

    We simply map indices to lowercase characters, e.g. 0 -> 'a', 1 -> 'b'.
    """
    return string.ascii_lowercase[i]


def unwrap(phase: KerasTensor, axis=-1) -> KerasTensor:
    nd = ops.ndim(phase)

    # get phase difference and correction
    phase_diff = ops.diff(phase, axis=axis)
    jumps = ops.cast(phase_diff < -pi, dtype="int8") - ops.cast(phase_diff > pi, dtype="int8")
    correction = ops.cumsum(ops.cast(jumps, dtype="float32") * 2.0 * pi, axis=axis)

    # pad to original size
    pad_width = [(0, 0)] * nd
    pad_width[axis] = (1, 0)

    correction = ops.pad(correction, pad_width=tuple(pad_width))
    return phase + correction


def capitalize_first_char(s: str):
    """
    Capitalize first character of string and leave the rest.
    

    Parameters
    ----------
    s : str
        A string to process.

    Returns
    -------
    s : str
        A modified string where the initial character is capitalized and the rest remains original.

    Notes
    -----
    Beneficial to import layers, e.g., "Conv2D".capitalize() results in "Conv2d", which is no valid layer!
    Source: https://stackoverflow.com/questions/12410242/python-capitalize-first-letter-only
    
    """

    return re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(), s, 1)


@saving.register_keras_serializable(package="KerasAddon.Helper", name="get_layer")
def get_layer(identifier, module="keras.layers", registered_name=None, **layer_kwargs):
    """
    Get a layer from an identifier

    Parameters
    ----------
    identifier : dict | str | Layer
        The identifier for a layer to return.
    module : str, optional
        The module to load the layer from.
        When loading official Keras layer, this should be `"keras.layers"`.
        When using a custom package, this should be the path to this package, e.g., `"keras_addon.layers"`.
        Defauts to `"keras.layers"`.
    registered_name : str, optional
        Only necessary if loading inofficial keras layers. Defaults to `None`.

    Returns
    -------
    layer : Layer
        An instance of a keras.Layer class from the identifier.

    """

    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = layers.deserialize(identifier)
    elif isinstance(identifier, str):
        config = {
            "module": module,
            "class_name": str(capitalize_first_char(identifier)),  # layer names are all capital!
            "config": {
                "name": None,
                "trainable": True,
                "dtype": {
                    "module": "keras",
                    "class_name": "DTypePolicy",
                    "config": {"name": "float32"},
                    "registered_name": None
                },
                **layer_kwargs
            },
            "registered_name": registered_name
        }
        obj = layers.deserialize(config)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret layer identifier: {identifier}"
        )
    