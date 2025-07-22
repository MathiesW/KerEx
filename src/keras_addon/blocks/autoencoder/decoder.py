from .base_decoder import BaseDecoder
from keras import saving


@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Decoder1D")
class Decoder1D(BaseDecoder):
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
            merge_layer="concatenate",
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
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
            merge_layer=merge_layer,
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
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
    

@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Decoder2D")
class Decoder2D(BaseDecoder):
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
            merge_layer="concatenate",
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
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
            merge_layer=merge_layer,
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
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


@saving.register_keras_serializable(package="KerasAddon.Blocks.Autoencoder", name="Decoder3D")
class Decoder3D(BaseDecoder):
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
            merge_layer="concatenate",
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            activity_regularizer=None, 
            trainable=True, dtype=None, 
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
            merge_layer=merge_layer, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
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
