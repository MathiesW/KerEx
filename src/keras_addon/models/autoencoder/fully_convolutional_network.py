from keras import saving
from .base_models import BaseFCN


@saving.register_keras_serializable(package="KerasAddon.Models.AutoEncoder", name="FCN1D")
class FCN1D(BaseFCN):
    def __init__(
        self,
        filters=[8, 16, 32],
        kernel_size=[5, 5, 5],
        data_format="channels_last",
        activation="relu",
        use_bias=True,
        bottleneck=None,
        merge_layer="concatenate",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bottleneck=bottleneck,
            merge_layer=merge_layer, 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")

        return config


@saving.register_keras_serializable(package="KerasAddon.Models.AutoEncoder", name="FCN2D")
class FCN2D(BaseFCN):
    def __init__(
        self,
        filters=[8, 16, 32],
        kernel_size=[5, 5, 5],
        data_format="channels_last",
        activation="relu",
        use_bias=True,
        bottleneck=None,
        merge_layer="concatenate",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bottleneck=bottleneck,
            merge_layer=merge_layer, 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")

        return config


@saving.register_keras_serializable(package="KerasAddon.Models.AutoEncoder", name="FCN3D")
class FCN3D(BaseFCN):
    def __init__(
        self,
        filters=[8, 16, 32],
        kernel_size=[5, 5, 5],
        data_format="channels_last",
        activation="relu",
        use_bias=True,
        bottleneck=None,
        merge_layer="concatenate",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bottleneck=bottleneck,
            merge_layer=merge_layer, 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.pop("rank")

        return config
    