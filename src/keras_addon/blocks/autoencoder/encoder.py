from .base_encoder import BaseEncoder
from keras import saving


@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Encoder1D")
class Encoder1D(BaseEncoder):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            data_format="channels_last", 
            dilation_rate=1, 
            groups=1, 
            activation="relu", 
            use_bias=True, 
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            use_skip_connection = False, 
            activity_regularizer=None, 
            trainable=True, dtype=None, 
            autocast=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=1, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            data_format=data_format, 
            dilation_rate=dilation_rate, 
            groups=groups, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            use_skip_connection=use_skip_connection, 
            activity_regularizer=activity_regularizer, 
            trainable=trainable, 
            dtype=dtype, 
            autocast=autocast, 
            name=name, 
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")
        return config
    

@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Encoder2D")
class Encoder2D(BaseEncoder):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            data_format="channels_last", 
            dilation_rate=1, 
            groups=1, 
            activation="relu", 
            use_bias=True, 
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            use_skip_connection = False, 
            activity_regularizer=None, 
            trainable=True, dtype=None, 
            autocast=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=2, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            data_format=data_format, 
            dilation_rate=dilation_rate, 
            groups=groups, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            use_skip_connection=use_skip_connection, 
            activity_regularizer=activity_regularizer, 
            trainable=trainable, 
            dtype=dtype, 
            autocast=autocast, 
            name=name, 
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Encoder3D")
class Encoder3D(BaseEncoder):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            data_format="channels_last", 
            dilation_rate=1, 
            groups=1, 
            activation="relu", 
            use_bias=True, 
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            use_skip_connection = False, 
            activity_regularizer=None, 
            trainable=True,
            dtype=None, 
            autocast=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=3, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            data_format=data_format, 
            dilation_rate=dilation_rate, 
            groups=groups, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            use_skip_connection=use_skip_connection, 
            activity_regularizer=activity_regularizer, 
            trainable=trainable, 
            dtype=dtype, 
            autocast=autocast, 
            name=name, 
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")
        return config
