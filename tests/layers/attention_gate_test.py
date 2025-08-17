from kerex.layers import AttentionGate
from keras import ops
import pytest


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_output_shape(dim):
    # get input data `x` forward and `s` skip info
    # skip info is double in size and half in channels
    x = ops.ones((1, *[8] * dim, 64))
    s = ops.ones((1, *[16] * dim, 32))

    attention_gate = AttentionGate()
    attention_gate.build(input_shape=[tuple(x.shape), tuple(s.shape)])

    y = attention_gate([x, s])
    target_shape = attention_gate.compute_output_shape(input_shape=[x.shape, s.shape])

    assert y.shape == target_shape
