from .base_fft_upsampling import BaseFFTUpSampling
from keras import saving


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling1D")
class FFTUpSampling1D(BaseFFTUpSampling):
    """
    1-D UpSampling in Forier space

    The input array is transformed via discrete Fourier transform for real-valued signals,
    padded with zeros, and
    transformed to physical space using inverse discrete Fourier transform for real-valued signals.

    Parameters
    ----------
    size : int | tuple
        Determines the upsampling factor between the input and output signals.
        Must be a positive integer or a tuple of positive integers.
        Defaults to `None`.
    target_size : int | tuple
        Determines the target size of the output signal.
        Can be smaller than initial input size to perform downsampling in Fourier space.
        This is an optional way to the `size` argument to determine the output size.
        Only one must be defined, either `size` or `target_size`.
        Defaults to `None`.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Format of the data.
        If `None`, this is usually `"channels_last"`, check Keras API!
        Defaults to `None`.
    name : str, optional
        Name of the layer.
        If `None`, it is derived from the class name.
        Defaults to `None`.
    **kwargs

    Raises
    ------
    ValueError
        When both `size` and `target_size` are not defined.
    ValueError
        When both `size` and `target_size` are defined.
    
    """

    def __init__(
            self, 
            size=None,
            target_size=None, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=1, 
            size=size, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling2D")
class FFTUpSampling2D(BaseFFTUpSampling):
    """
    2-D UpSampling in Forier space

    The input array is transformed via discrete Fourier transform for real-valued signals,
    padded with zeros, and
    transformed to physical space using inverse discrete Fourier transform for real-valued signals.

    Parameters
    ----------
    size : int | tuple
        Determines the upsampling factor between the input and output signals.
        Must be a positive integer or a tuple of positive integers.
        Defaults to `None`.
    target_size : int | tuple
        Determines the target size of the output signal.
        Can be smaller than initial input size to perform downsampling in Fourier space.
        This is an optional way to the `size` argument to determine the output size.
        Only one must be defined, either `size` or `target_size`.
        Defaults to `None`.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Format of the data.
        If `None`, this is usually `"channels_last"`, check Keras API!
        Defaults to `None`.
    name : str, optional
        Name of the layer.
        If `None`, it is derived from the class name.
        Defaults to `None`.
    **kwargs

    Raises
    ------
    ValueError
        When both `size` and `target_size` are not defined.
    ValueError
        When both `size` and `target_size` are defined.
    
    """

    def __init__(
            self, 
            size=None,
            target_size=None, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=2, 
            size=size, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling3D")
class FFTUpSampling3D(BaseFFTUpSampling):
    """
    2-D UpSampling in Forier space

    The input array is transformed via discrete Fourier transform for real-valued signals,
    padded with zeros, and
    transformed to physical space using inverse discrete Fourier transform for real-valued signals.

    Parameters
    ----------
    size : int | tuple
        Determines the upsampling factor between the input and output signals.
        Must be a positive integer or a tuple of positive integers.
        Defaults to `None`.
    target_size : int | tuple
        Determines the target size of the output signal.
        Can be smaller than initial input size to perform downsampling in Fourier space.
        This is an optional way to the `size` argument to determine the output size.
        Only one must be defined, either `size` or `target_size`.
        Defaults to `None`.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Format of the data.
        If `None`, this is usually `"channels_last"`, check Keras API!
        Defaults to `None`.
    name : str, optional
        Name of the layer.
        If `None`, it is derived from the class name.
        Defaults to `None`.
    **kwargs

    Raises
    ------
    ValueError
        When both `size` and `target_size` are not defined.
    ValueError
        When both `size` and `target_size` are defined.
    
    """

    def __init__(
            self, 
            size=None,
            target_size=None, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=3, 
            size=size, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )
