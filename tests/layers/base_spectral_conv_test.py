from kerex.layers.fno.spectral_conv.base_spectral_conv import BaseSpectralConv
from keras import ops, random
import pytest


DEFAULT_FILTERS = 1
DEFAULT_MODES = 4


def array_equal(x1, x2):
    return ops.convert_to_numpy(ops.equal(x1, x2)).all()


def get_data(rank):
    n = ops.prod([1, *[2*DEFAULT_MODES] * rank, DEFAULT_FILTERS])
    x = ops.reshape(ops.arange(n) - n // 2, (1, *[2*DEFAULT_MODES] * rank, DEFAULT_FILTERS))
    x = ops.cast(x, dtype="float32")
    x += random.normal(shape=x.shape)

    return x


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_forward(rank):
    x = get_data(rank=rank)
    layer = BaseSpectralConv(rank=rank, filters=DEFAULT_FILTERS, modes=DEFAULT_MODES)

    layer.build(x.shape)

    # simulate forward path (using only the real part)
    xf, _ = layer.rfft(x)
    xf_shifted = layer.truncation_shift(xf)
    xf_truncated = xf_shifted[layer.mode_truncation_slice]  # zero mode should always maintain!
    yf_truncated = ops.einsum(layer.einsum_op_forward, xf_truncated, layer._real_kernel)  # neglect imag part here, we are only intersted in shapes and correct truncation / padding
    yf_padded = ops.pad(yf_truncated, pad_width=layer.pad_width)
    yf_shifted = layer.truncation_shift(yf_padded, inverse=True)
    y = layer.irfft((yf_shifted, yf_shifted))  # just use the real part twice here...

    assert x.shape == y.shape, f"Wrong output shape!"
    assert xf_truncated.shape == (1, 1, *layer.modes), f"Shape of truncated `x` deviates from `modes`"

    deviation = ops.sum(yf_truncated) - ops.sum(yf_padded)
    assert ops.abs(deviation) < 1e-3, f"Deviation between truncated and zero-padded tensor exceeds `1e-3` ({deviation})"


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_truncation_shift(rank):
    x = get_data(rank=rank)  # (1, 8, 8, 1)
    layer = BaseSpectralConv(rank=rank, filters=DEFAULT_FILTERS, modes=DEFAULT_MODES)

    layer.build(x.shape)
    
    x_transposed = layer.transpose(x)  # (1, 1, 8, 8)
    x_shifted = layer.truncation_shift(x_transposed)  # (1, 1, 8, 8)
    x_reconstructed = layer.truncation_shift(x_shifted, inverse=True)  # (1, 1, 8, 8)
    x_reconstructed = layer.inverse_transpose(x_reconstructed)  # (1, 8, 8, 1)

    assert array_equal(x, x_reconstructed), f"Truncation shift does not reconstruct original tensor"
