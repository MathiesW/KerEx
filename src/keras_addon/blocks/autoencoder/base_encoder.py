from keras import layers
from ...layers.conv import BaseConv


class BaseEncoder(layers.Layer):
    """ BaseEncoder
    Downsampling is realized using a strided convolution

    Feel free to subclass and change the layers!
    """
    def __init__(
            self,
            rank,
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
            use_skip_connection: bool = False,
            activity_regularizer=None, 
            trainable=True, 
            dtype=None, 
            autocast=True, 
            name=None, 
            **kwargs):
        super().__init__(
            activity_regularizer=activity_regularizer, 
            trainable=trainable, 
            dtype=dtype, 
            autocast=autocast, 
            name=name, 
            **kwargs
        )

        self.rank = rank
        self.use_skip_connection = use_skip_connection

        # define layers
        self.forward_conv = BaseConv(
            rank=self.rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )
        self.downsampling = BaseConv(
            rank=self.rank,
            filters=filters,
            kernel_size=2,
            strides=2,
            padding="same",
            data_format=data_format,
            dilation_rate=1,
            groups=groups,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

    def call(self, inputs):
        """
        
        Parameters
        ----------
        inputs : KerasTensor
            ...
        skip : KerasTensor, optional
            ...
            Defaults to `None`.

        Returns
        -------
        outputs | (outputs_forward, outputs_skip) : KerasTensor | (KerasTensor, KerasTensor)
            ...
            
        """

        x_skip = self.forward_conv(inputs)
        x_forward = self.downsampling(x_skip)

        if self.use_skip_connection:
            return x_forward, x_skip
        
        return x_forward
    
    def build(self, input_shape: tuple):
        if self.built:
            return
        
        super().build(input_shape)

        # build forward layer and update input_shape
        self.forward_conv.build(input_shape=input_shape)
        input_shape = self.forward_conv.compute_output_shape(input_shape=input_shape)

        # build downsampling layer
        self.downsampling.build(input_shape=input_shape)

        # update built state
        self.built = True
    
    def compute_output_shape(self, input_shape: tuple) -> tuple:
        output_shape_skip = self.forward_conv.compute_output_shape(input_shape=input_shape)
        output_shape_forward = self.downsampling.compute_output_shape(input_shape=output_shape_skip)

        if self.use_skip_connection:
            return (output_shape_forward, output_shape_skip)
        
        return output_shape_forward

    def get_config(self) -> dict:
        config: dict = super().get_config()
        forward_config: dict = self.forward_conv.get_config()
        forward_config.pop("padding")
        
        return {**config, **forward_config, "rank": self.rank, "use_skip_connection": self.use_skip_connection}
    