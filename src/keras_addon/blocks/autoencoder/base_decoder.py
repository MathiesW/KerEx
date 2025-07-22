from keras import layers
from keras import saving
from ...layers.conv import BaseConv, BaseConvTranspose
from keras.src.layers.merging.base_merge import Merge
from ...ops import get_layer


class BaseDecoder(layers.Layer):
    """ BaseDecoder
    Upsampling is realized using a transposed convolution

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
            merge_layer='concatenate',
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
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
        self.upsampling = BaseConvTranspose(
            rank=self.rank,
            filters=filters,
            kernel_size=2,
            strides=2,
            padding="same",
            data_format=data_format,
            dilation_rate=1,
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

        # load merge layer
        try:
            self.merge_layer = get_layer(merge_layer, axis=-1 if data_format == "channels_last" else 1)
        except TypeError:  # layer does not supply axis argument
            self.merge_layer = get_layer(merge_layer)

        if not issubclass(type(self.merge_layer), Merge):
            raise TypeError(
                f"Merge-layer {self.merge_layer} supplied to Wrapper isn't "
                "a supported merge-layer."
            )
        
    def call(self, inputs, skip=None):
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
        outputs : KerasTensor
            ...
            
        """

        x_forward = self.upsampling(inputs)

        if skip is not None:
            x_forward = self.merge_layer((x_forward, skip))

        x_forward = self.forward_conv(x_forward)

        return x_forward
    
    def build(self, input_shape: tuple, input_shape_skip: tuple = None):
        if self.built:
            return
        
        # cache input shapes
        # self._build_shapes_dict = {"input_shape": input_shape, "input_shape_skip": input_shape_skip}
        self._build_shapes = {}

        super().build(input_shape)

        # build upsampling layer
        # cache build shape
        self._build_shapes.update({self.upsampling.name: {"input_shape": input_shape}})

        self.upsampling.build(input_shape=input_shape)
        input_shape = self.upsampling.compute_output_shape(input_shape=input_shape)

        # build merge layer
        if input_shape_skip is not None:
            # cache build shape
            self._build_shapes.update({self.merge_layer.name: {"input_shape": (input_shape, input_shape_skip)}})

            self.merge_layer.build(input_shape=(input_shape, input_shape_skip))
            input_shape = self.merge_layer.compute_output_shape(input_shape=(input_shape, input_shape_skip))

        # build forward layer and update input_shape
        # cache build shape
        self._build_shapes.update({self.forward_conv.name: {"input_shape": input_shape}})

        self.forward_conv.build(input_shape=input_shape)

        # update built state
        self.built = True
    
    def compute_output_shape(self, input_shape: tuple, input_shape_skip: tuple = None) -> tuple:
        input_shape = self.upsampling.compute_output_shape(input_shape=input_shape)
        if input_shape_skip is not None:
            input_shape = self.merge_layer.compute_output_shape(input_shape=(input_shape, input_shape_skip))

        output_shape = self.forward_conv.compute_output_shape(input_shape=input_shape)
        
        return output_shape

    def get_config(self) -> dict:
        config: dict = super().get_config()
        forward_config: dict = self.forward_conv.get_config()
        forward_config.pop("padding")

        config.update({"merge_layer": saving.serialize_keras_object(self.merge_layer)})
        
        return {**config, **forward_config, "rank": self.rank}
    
    @classmethod
    def from_config(cls, config: dict):
        merge_cfg = config.pop("merge_layer")
        config.update({"merge_layer": saving.deserialize_keras_object(merge_cfg)})
        
        return cls(**config)
