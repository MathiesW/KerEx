from keras_addon.models.autoencoder.base_models import BaseFCN as BaseModel
from keras import ops
from keras.src.backend.config import backend
import pytest


DEFAULT_FILTERS = [8, 16, 32]


@pytest.mark.parametrize("padding", ["same", "causal", ["same", "causal", "same"]])
def test_padding_modes_1d(padding):
    BaseModel(rank=1, padding=padding)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_filters_as_tuple(rank):
    filters = [(8, 8), 16, 32]
    BaseModel(rank=rank, filters=filters)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_kernel_size_as_tuple(rank):
    kernel_size = [tuple([2 * rank] * rank)] * len(DEFAULT_FILTERS)
    BaseModel(rank=rank, filters=DEFAULT_FILTERS, kernel_size=kernel_size)


@pytest.mark.parametrize("merge_layer", ["concatenate", "add", "average"])
def test_merge_layers(merge_layer):
    x = ops.ones(tuple([1, 16, 3]))
    model = BaseModel(rank=1, merge_layer=merge_layer, use_skip_connection=True)
    model.build(input_shape=x.shape)

    model(x)


""" check for error raising """
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_padding_mode_valid_raise_ValueError(rank):
    with pytest.raises(ValueError):
        BaseModel(rank=rank, padding="valid")


def test_rank_higher_than_three():
    x = ops.ones((1, 32, 32, 32, 32, 3))
    model = BaseModel(rank=4)
    model.build(input_shape=x.shape)

    if backend == "jax":
        # obviously, 4-D+ convolutions are possible with JAX backend
        model(x)
    
    if backend == "tensorflow":
        # ...but not with Tensorflow backend!
        with pytest.raises(ValueError):
            model(x)


""" serialization """
...


""" long tests """
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_output_shape_is_correct(rank):
    x = ops.ones(tuple([1, *[16] * rank, 3]))

    model = BaseModel(rank=rank)
    model.build(input_shape=x.shape)

    expected_output_shape = model.compute_output_shape(input_shape=x.shape)

    y = model(x)
    actual_output_shape = y.shape

    assert expected_output_shape == actual_output_shape, f"Expected output shape ({expected_output_shape}) deviates from actual output shape ({actual_output_shape})!"


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_partial_skip_connection(rank):
    x = ops.ones(tuple([1, *[16] * rank, 3]))

    model = BaseModel(rank=rank, use_skip_connection=[False, False, True])
    model.build(input_shape=x.shape)

    model(x)