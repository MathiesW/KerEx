from kerex.models.neural_operator.base_neural_operator import BaseNeuralOperator as BaseModel
from keras import ops
import pytest


DEFAULT_FILTERS = [8, 8, 8]
DEFAULT_MODES = 4


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_modes_as_tuple(rank):
    modes = [tuple([DEFAULT_MODES] * rank)] * len(DEFAULT_FILTERS)
    BaseModel(rank=rank, filters=DEFAULT_FILTERS, modes=modes)


@pytest.mark.parametrize("merge_layer", ["concatenate", "add", "average"])
def test_merge_layers(merge_layer):
    x = ops.ones((1, 16, 3))
    model = BaseModel(rank=1, filters=DEFAULT_FILTERS, modes=DEFAULT_MODES, merge_layer=merge_layer)
    model.build(input_shape=x.shape)

    model(x)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_data_formats(data_format):
    x = ops.ones((1, 16, 3) if data_format == "channels_last" else (1, 3, 16))
    model = BaseModel(rank=1, filters=DEFAULT_FILTERS, modes=DEFAULT_MODES, data_format=data_format)
    model.build(input_shape=x.shape)

    model(x)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_output_shape_is_correct(rank):
    x = ops.ones(tuple([1, *[16] * rank, 3]))

    model = BaseModel(rank=rank, filters=DEFAULT_FILTERS, modes=DEFAULT_MODES)
    model.build(input_shape=x.shape)

    expected_output_shape = model.compute_output_shape(input_shape=x.shape)

    y = model(x)
    actual_output_shape = y.shape

    assert expected_output_shape == actual_output_shape, f"Expected output shape ({expected_output_shape}) deviates from actual output shape ({actual_output_shape})!"


""" check for error raising """
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_zero_modes_raise_ValueError(rank):
    with pytest.raises(ValueError):
        BaseModel(rank=rank, filters=DEFAULT_FILTERS, modes=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_modes_greater_data_raise_ValueError(rank):
    size = 16

    x = ops.ones((1, *[size] * rank, 3))
    model = BaseModel(rank=rank, filters=DEFAULT_FILTERS, modes=size // 2 + 2)

    with pytest.raises(ValueError):
        model.build(x.shape)


# """ training behavior """
# @pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
# def test_backprop(data_format):
#     x = ops.ones((1, 16, 16, 3) if data_format == "channels_last" else (1, 3, 16, 16))
#     y = ops.ones((1, 16, 16, 1) if data_format == "channels_last" else (1, 1, 16, 16))

#     model = BaseModel(rank=2, filters=4, modes=4, output_projection_dimension=1, data_format=data_format)
#     model.build(input_shape=x.shape)
#     model.compile(optimizer="adam", loss="mse")

#     model.fit(x=x, y=y, epochs=10, batch_size=1)
