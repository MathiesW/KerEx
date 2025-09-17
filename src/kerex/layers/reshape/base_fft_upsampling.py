from keras import layers
from importlib import import_module
from functools import partial
from keras import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.backend import standardize_data_format
from keras.src.utils.argument_validation import standardize_tuple


class BaseFFTUpSampling(layers.Layer):
    """
    UpSampling in Forier space

    The input array is transformed via discrete Fourier transform for real-valued signals,
    padded with zeros, and
    transformed to physical space using inverse discrete Fourier transform for real-valued signals.

    Parameters
    ----------
    rank : int
        Rank of the data.
        Determines rank of Fourier transform.
    target_size : int | tuple
        Determines the target size of the output signal.
        Can be smaller than initial input size to perform downsampling in Fourier space.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Format of the data.
        If `None`, this is usually `"channels_last"`, check Keras API!
        Defaults to `None`.
    name : str, optional
        Name of the layer.
        If `None`, it is derived from the class name.
        Defaults to `None`.
    **kwargs
    
    """

    def __init__(
        self, 
        rank,
        target_size,
        data_format=None,
        name=None, 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rank = rank
        
        self.target_size = standardize_tuple(target_size, n=self.rank, allow_zero=True, name="size")
        self.data_format = standardize_data_format(data_format)

        # import fft 
        self.rfft_fn = getattr(import_module(name="keras_fft", package=__package__), "rfft" if self.rank == 1 else f"rfft{self.rank}")
        self.irfft_fn = getattr(import_module(name="keras_fft", package=__package__), f"irfft" if self.rank == 1 else f"irfft{self.rank}")
        
    def build(self, input_shape):
        if self.built:
            return

        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]

            # get data axes
            axes = list(range(len(input_shape)))

            # if data format is `"channels_last"`, we have to transpose in order to apply the rfft and irfft along the last axes
            transpose_axes = axes.copy()
            inverse_transpose_axes = axes.copy()

            transpose_axes.insert(1, transpose_axes.pop())
            inverse_transpose_axes.append(inverse_transpose_axes.pop(1))

            self.transpose = partial(ops.transpose, axes=transpose_axes)
            self.inverse_transpose = partial(ops.transpose, axes=inverse_transpose_axes)

        else:
            # NOTE this does not matter too much since currently the layer is restricted to use `"channels_last"` data format!
            channel_axis = 1
            input_channel = input_shape[1]

            # if data format is already `"channels_first"`, we do not have to transpose in order to apply the rfft and irfft
            self.transpose = lambda x: x
            self.inverse_transpose = lambda x: x

        # define input spec
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )

        # define pad with. In Fourier space, we will always have `data_format="channels_first"`!
        # size is a tuple of positive integers that act as a multiplier for the size in each dimension
        # we use RFFT -> shape in Fourier space is 
        # target_shape_fft = target_shape_ph // 2 + 1
        # target_shape_ph = (target_shape_fft + 1) * 2
        # target_shape_fft = initial_shape_fft + padding
        # initial_shape_fft = input_shape or input_shape // 2 + 1 (along last axis when using RFFT)
        # padding   = target_shape_fft - initial_shape_fft
        #           = target_shape_ph - initial_shape_ph                        if axis != -1
        #           = (target_shape_ph // 2 + 1) - (initial_shape_ph // 2 + 1)  else

        # get initial shape along feature axes
        feature_axes = list(range(len(input_shape)))
        feature_axes.pop(channel_axis)  # remove channel axis
        feature_axes.pop(0)  # remove batch dimension
        feature_dims = tuple([input_shape[fa] for fa in feature_axes])

        # Parcivals Theorem: constant energy in frequency spectrum. We have to scale output after padding!
        self.scaling_factor = ops.prod(self.target_size) / ops.prod(feature_dims)

        self.pad_width = (
            (0, 0),
            (0, 0), 
            *[(0, (size // 2 + 1) - (shape // 2 + 1)) if (i == (len(self.target_size) - 1)) else ((size - shape) // 2, (size - shape) // 2) for i, (size, shape) in enumerate(zip(self.target_size, input_shape[(1 if self.data_format == "channels_last" else 2):]))]
        )

        self.built = True

    def rfft(self, x):
        """
        Performs fast Fourier transform on the real-valued `inputs`.
        
        Parameters
        ----------
        inputs : KerasTensor
            Real-valued input tensor.
        
        Returns
        -------
        (y_real, y_imag) : (KerasTensor, KerasTensor)
            Real- and imaginary part of fast Fourier tranform applied to `inputs`.

        Notes
        -----
        `inputs` must be of shape `(batch, channels, *features)`.

        """
        
        x = self.transpose(x)
        x_real, x_imag = self.rfft_fn(x)

        return x_real, x_imag
    
    def irfft(self, inputs):
        """
        Performs inverse fast Fourier transform on the `inputs`.
        
        Parameters
        ----------
        inputs : tuple
            Tuple of `KerasTensor` (real and imaginary part).
        
        Returns
        -------
        outputs : KerasTensor
            Real-valued output of inverse fast Fourier transform applied to `inputs`.

        Notes
        -----
        `inputs` must be of shape `((batch, channels, *features), (batch, channels, *features))`.

        """

        x_real, x_imag = inputs
        y = self.irfft_fn((x_real, x_imag))
        
        return self.inverse_transpose(y)

    # NOTE: this shouldn't be necessary since there are no trainable weights! MAYBE to get FFT backprop correct in TF?
    # def call(self, inputs):
    #     if backend() == "tensorflow":
    #         @ops.custom_gradient
    #         def forward(inputs):
    #             x_real, x_imag = self.rfft(inputs)
                
    #             x_real_padded = ops.pad(x_real, pad_width=self.pad_width)
    #             x_imag_padded = ops.pad(x_imag, pad_width=self.pad_width)

    #             y = self.irfft(x_real_padded, x_imag_padded)

    #             def backprop(dy, variables=None):
    #                 dy_real, dy_imag = self.rfft(dy)

    #                 dy_real_truncated = dy_real[self.truncation_slice]
    #                 dy_imag_truncated = dy_imag[self.truncation_slice]

    #                 dx = self.irfft(dy_real_truncated, dy_imag_truncated)

    #                 return dx, []
                
    #             return y, backprop
            
    #         return forward(inputs)
        
    #     if backend() == "jax":
    #         x_real, x_imag = self.rfft(inputs)
                
    #         x_real_padded = ops.pad(x_real, pad_width=self.pad_width)
    #         x_imag_padded = ops.pad(x_imag, pad_width=self.pad_width)

    #         y = self.irfft(x_real_padded, x_imag_padded)

    #         return y
    
    #     raise NotImplementedError(f"The call method is only implemented for keras backends `'tensorflow'` and `'jax'`")

    def call(self, inputs):
        x_real, x_imag = self.rfft(inputs)
                
        x_real_padded = ops.pad(x_real, pad_width=self.pad_width)
        x_imag_padded = ops.pad(x_imag, pad_width=self.pad_width)

        y = self.irfft((x_real_padded, x_imag_padded))

        # scale the signal
        y *= ops.cast(self.scaling_factor, dtype=y.dtype)

        return y

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)

        feature_axes = list(range(len(input_shape)))
        feature_axes.pop(0)  # batch
        feature_axes.pop(-1 if self.data_format == "channels_last" else 0)  # channel dimension

        for fa, size in zip(feature_axes, self.target_size):
            input_shape[fa] = size

        return tuple(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "target_size": self.target_size,
            "data_format": self.data_format
        })

        return config
