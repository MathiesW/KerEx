from keras import Layer, KerasTensor
from keras import regularizers, initializers
from keras import ops
from typing import Tuple
from ...ops.fft import rfft, rfft2, irfft, irfft2
from functools import partial


class BaseSpectralConv1D(Layer):
    def __init__(
        self, 
        filters: int, 
        modes: int, 
        name: str = None, 
        kernel_regularizer: regularizers.Regularizer = None, 
        bias_regularizer: regularizers.Regularizer = None, 
        activity_regularizer: regularizers.Regularizer = None, 
        data_format: str = 'channels_last', 
        use_bias: bool = True,
        **kwargs
    ):
        # NVIDIA recommends 'using the NHWC format where possible': https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
        super().__init__(
            name=name, 
            activity_regularizer=activity_regularizer, 
            **kwargs
        )
        self.data_format = data_format.lower()
        assert self.data_format in ['channels_first', 'channels_last']
        if self.data_format == 'channels_first':
            raise NotImplementedError(f"Data format 'channels_first' is currently not supported.\nNVIDIA recommends to use 'channels_last' anyway!")

        self.filters = filters
        self.modes = modes
        self.data_axes = (-2,)  # for convolution with channels last
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.use_bias = use_bias

        # callables
        self.transpose_to_channels_first = partial(ops.transpose, axes=[0, 1, 2] if data_format =='channels_first' else [0, 2, 1])
        self.inverse_transpose = partial(ops.transpose, axes=[0, 1, 2] if data_format =='channels_first' else [0, 2, 1])
        self.rfft_fn = rfft
        self.irfft_fn = irfft
        self.rfft_scaling = None  # is determined during build routine!

    def build(self, input_shape: tuple) -> None:
        if self.built:
            return
        
        modes = (self.modes, ) if isinstance(self.modes, int) else tuple(self.modes)  # work with tuple for dimensions

        # get scaling factor for rfft and irfft operations
        feature_dimension = [input_shape[a] for a in self.data_axes]
        self.rfft_scaling = ops.cast(ops.prod(feature_dimension), dtype=self.dtype) / 2.0

        # get scaling for weight initialization
        in_channels = input_shape[-1]
        scale = 1.0 / (in_channels * self.filters)  # convention due to original paper

        self.real_weights = self.add_weight(
            shape=(in_channels, *modes, self.filters),
            initializer=initializers.HeNormal(),
            # initializer=initializers.RandomUniform(minval=-scale, maxval=scale),
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='real_weights'
        )
        self.imag_weights = self.add_weight(
            shape=(in_channels, *modes, self.filters),
            initializer=initializers.HeNormal(),
            # initializer=initializers.RandomUniform(minval=-scale, maxval=scale),
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='imag_weights'
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters, ), 
                initializer=initializers.Zeros(),
                regularizer=self.bias_regularizer,
                trainable=True, 
                name='bias'
            )

        self.pad_width = (
            (0, 0),
            *[(0, s // 2 + 1 - m if i == (len(modes) - 1) else s - m) for i, (m, s) in enumerate(zip(modes, input_shape[1:]))],
            (0, 0)
        )

        if list(filter(lambda x: x < (0, 0), self.pad_width)):
                raise ValueError("Too many modes for input shape!")

        # declare einsum operation
        einsum_dim = ''.join([d for _, d in zip(modes, ['X', 'Y', 'Z'])])
        self.einsum_op_forward = f'b{einsum_dim}i,i{einsum_dim}o->b{einsum_dim}o'
        self.einsum_op_backprop_weights = f'b{einsum_dim}o,b{einsum_dim}i->i{einsum_dim}o'
        self.einsum_op_backprop_x = f'b{einsum_dim}o,i{einsum_dim}o->b{einsum_dim}i'
        self.einsum_op_backprop_bias = f'b{einsum_dim}o->o'  # sum over all axis except output channels

        self.mode_truncation_slice = tuple([slice(None), *[slice(None, m) for m in modes]])


        self.built = True

    def rfft(self, x: KerasTensor) -> Tuple[KerasTensor, KerasTensor]:
        x = self.transpose_to_channels_first(x)
        x_real, x_imag = self.rfft_fn(x)

        # scale outputs for numerical stability in Fourier space
        x_real /= self.rfft_scaling
        x_imag /= self.rfft_scaling

        return self.inverse_transpose(x_real), self.inverse_transpose(x_imag)
    
    def irfft(self, x: Tuple[KerasTensor, KerasTensor]) -> KerasTensor:
        x_real, x_imag = x
        x_real = self.transpose_to_channels_first(x_real)
        x_imag = self.transpose_to_channels_first(x_imag)
        y_real = self.irfft_fn((x_real, x_imag))

        # scale back to "normal" scale
        y_real *= self.rfft_scaling
        return self.inverse_transpose(y_real)

    def call(self, x: KerasTensor):
        NotImplemented

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        input_shape: list = list(input_shape)
        channel_axis = -1 if self.data_format == 'channels_last' else 1

        input_shape[channel_axis] = self.filters
        return tuple(input_shape)

    def get_config(self) -> dict:
        # https://keras.io/guides/serialization_and_saving/
        config: dict = super().get_config()
        config.update({
            'filters': self.filters,
            'modes': self.modes,
            'data_format': self.data_format,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        })
        return config
    

class BaseSpectralConv2D(BaseSpectralConv1D):
    def __init__(
        self, 
        filters: int, 
        modes: int = None, 
        modes_x: int = None, 
        modes_y: int = None, 
        name: str = None, 
        kernel_regularizer: regularizers.Regularizer = None, 
        bias_regularizer: regularizers.Regularizer = None, 
        activity_regularizer: regularizers.Regularizer = None, 
        data_format: str = 'channels_last', 
        **kwargs
    ):
        super().__init__(
            filters=filters, 
            modes=modes,
            name=name, 
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            data_format=data_format, 
            **kwargs
        )
        if (self.modes is None) & (modes_x is None) & (modes_y is None):
            raise ValueError(f"Not all modes can be None. Please define either 'modes' (global) or 'modes_x' and 'modes_y'")
        
        # overwrite self.modes
        self.modes = (modes_y or modes, modes_x or modes)  # for me, x is the 'shorter' dimension after rfft --> not intuitive!
        self.data_axes = (-3, -2)  # for convolution with channels last

        # callables
        self.transpose_to_channels_first = partial(ops.transpose, axes=[0, 1, 2, 3] if data_format =='channels_first' else [0, 3, 1, 2])
        self.inverse_transpose = partial(ops.transpose, axes=[0, 1, 2, 3] if data_format =='channels_first' else [0, 2, 3, 1])
        self.rfft_fn = rfft2
        self.irfft_fn = irfft2

    def build(self, input_shape: Tuple) -> None:
        super().build(input_shape)

        # remove deprecated class attributes from 1D implementation
        delattr(self, 'mode_truncation_slice')
        delattr(self, 'pad_width')
        
        modes_y, modes_x = self.modes

        if not modes_y % 2 == 0:
            raise ValueError(f"Odd number of y-modes is currently not supported by {self.name}, received y_modes={modes_y}.")

        # truncation
        self.mode_truncation_slice_pos = tuple([slice(None), slice(None, modes_y // 2), slice(None, modes_x)])
        self.mode_truncation_slice_neg = tuple([slice(None), slice(-(modes_y // 2), None), slice(None, modes_x)])

        # padding
        self.pad_width_pos = (
            (0, 0),
            (0, (input_shape[1] - modes_y) // 2),
            (0, input_shape[2] // 2 + 1 - modes_x),
            (0, 0)
        )
        self.pad_width_neg = (
            (0, 0),
            ((input_shape[1] - modes_y) // 2, 0),
            (0, input_shape[2] // 2 + 1 - modes_x),
            (0, 0)
        )

        for pw in [self.pad_width_pos, self.pad_width_neg]:
            if list(filter(lambda x: x < (0, 0), pw)):
                raise ValueError("Too many modes for input shape!")

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update({
            'modes_x': self.modes[1],
            'modes_y': self.modes[0],
        })
        return config
