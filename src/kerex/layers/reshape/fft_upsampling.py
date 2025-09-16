from .base_fft_upsampling import BaseFFTUpSampling
from keras import saving


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling1D")
class FFTUpSampling1D(BaseFFTUpSampling):
    def __init__(
            self, 
            target_size, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=1, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling2D")
class FFTUpSampling2D(BaseFFTUpSampling):
    def __init__(
            self, 
            target_size, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=2, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Reshape", name="FFTUpSampling3D")
class FFTUpSampling3D(BaseFFTUpSampling):
    def __init__(
            self, 
            target_size, 
            data_format=None, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=3, 
            target_size=target_size, 
            data_format=data_format, 
            name=name, 
            **kwargs
        )