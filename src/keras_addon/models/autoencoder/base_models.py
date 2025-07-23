from keras import Model
from keras import regularizers, initializers, constraints
from keras import saving
from ...blocks.autoencoder.base_encoder import BaseEncoder
from ...blocks.autoencoder.base_decoder import BaseDecoder
from ...blocks.autoencoder.base_block import _BaseConvBlock


class BaseFCN(Model, _BaseConvBlock):
    """
    Base class of FullyConvolutionalNetwork (FCN)

    Convolutional autoencoder *without* skip-connections

    Notes
    -----
    Downsampling is realized using a strided convolution as proposed by [Springenberger et al.](https://arxiv.org/abs/1412.6806)

    """

    def __init__(
            self,
            rank,
            filters=[8, 16, 32],
            kernel_size=5,
            strides=1,
            padding="same",  # only 1-D implementation may use `"causal"` padding instead of `"same"` padding!
            data_format="channels_last",
            dilation_rate=1,
            groups=1,
            use_skip_connection=False,
            activation="relu",
            use_bias=True,
            bottleneck=None,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs
        ):
        
        self.merge_layer = kwargs.pop("merge_layer", "concatenate")
        super().__init__(name=name, **kwargs)

        self.rank = rank
        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # set iterable class variables for the model
        self.set_vars(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_skip_connection=use_skip_connection
        )

        # check `padding` mode
        if (self.rank != 1) & any(item != "same" for item in self.padding):
            raise ValueError(f"For `rank={self.rank}`, `'same'` is the only valid padding mode, received `padding={self.padding}`.")
        
        if (self.rank == 1) & any(item not in ["same", "causal"] for item in self.padding):
            raise ValueError(f"Valid padding modes for `rank=1` are `'same'` and `'causal'`, received `padding={self.padding}`.")

        self.set_encoder_layers()
        self.bottleneck = bottleneck
        self.set_decoder_layers()

        # cache build shapes
        self.global_build_shapes_dict = None        

    def set_encoder_layers(self):
        self.encoder_layers = [
            BaseEncoder(
                rank=self.rank,
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                data_format=self.data_format,
                dilation_rate=d,
                groups=g,
                use_skip_connection=skip,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                activity_regularizer=self.activity_regularizer,
                name=f"Encoder_{i}"
            ) for i, (f, k, s, p, d, g, skip) in enumerate(
                zip(
                    self.filters,
                    self.kernel_size, 
                    self.strides,
                    self.padding, 
                    self.dilation_rate, 
                    self.groups, 
                    self.use_skip_connection
                )
            )
        ]
    
    def set_decoder_layers(self):
        self.decoder_layers = [
            BaseDecoder(
                rank=self.rank,
                filters=f,
                kernel_size=k,
                strides=s,
                padding=p,
                data_format=self.data_format,
                dilation_rate=d,
                groups=g,
                merge_layer=self.merge_layer, 
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                activity_regularizer=self.activity_regularizer,
                name=f"Decoder_{len(self.filters) - i - 1}"
            ) for i, (f, k, s, p, d, g) in enumerate(
                zip(
                    list(reversed(self.filters)), 
                    list(reversed(self.kernel_size)),
                    list(reversed(self.strides)),
                    list(reversed(self.padding)),
                    list(reversed(self.dilation_rate)),
                    list(reversed(self.groups))
                )
            )
        ]

    def build(self, input_shape: tuple):
        if self.built:
            return
        
        # cache build shapes
        self.global_build_shapes_dict = {}

        # build encoder layers
        output_shapes = []
        for layer, layer_uses_skip in zip(self.encoder_layers, self.use_skip_connection):
            # cache input shape
            self.global_build_shapes_dict.update({layer.name: {"input_shape": input_shape}})

            layer.build(input_shape=input_shape)

            # update input shape and append to output_shapes
            input_shape = layer.compute_output_shape(input_shape=input_shape)
            if layer_uses_skip:
                input_shape, input_shape_skip = input_shape
                output_shapes.append(input_shape_skip)
            else:
                output_shapes.append(None)

        # build bottleneck layers
        if self.bottleneck is not None:
            # cache input shape
            self.global_build_shapes_dict.update({self.bottleneck.name: {"input_shape": input_shape}})

            self.bottleneck.build(input_shape=input_shape)
            input_shape = self.bottleneck.compute_output_shape(input_shape=input_shape)

        # now build decoder layers
        output_shapes.reverse()

        for layer, input_shape_skip in zip(self.decoder_layers, output_shapes):
            # cache input shape
            self.global_build_shapes_dict.update({layer.name: {"input_shape": input_shape, "input_shape_skip": input_shape_skip}})

            layer.build(input_shape=input_shape, input_shape_skip=input_shape_skip)
            input_shape = layer.compute_output_shape(input_shape=input_shape, input_shape_skip=input_shape_skip)

        self.built = True

    def call(self, inputs):
        """
        
        Parameters
        ----------
        inputs : KerasTensor
            Input to BaseFCN

        Returns
        -------
        outputs : KerasTensor
            Output of BaseFCN

        """

        skip = []
        # forward path through encoder
        x = inputs
        for layer, layer_uses_skip in zip(self.encoder_layers, self.use_skip_connection):
            x = layer(x)
            if layer_uses_skip:
                x, x_skip = x
                skip.append(x_skip)
            else:
                skip.append(None)

        # apply bottlebeck layer
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        # forward path through decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, skip=skip[len(self.decoder_layers) - i - 1])

        return x
    
    def compute_output_shape(self, input_shape: tuple) -> tuple:
        output_shape = list(input_shape)
        output_shape[-1] = self.filters[0]

        return tuple(output_shape)

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update({
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "use_skip_connection": self.use_skip_connection,
            "merge_layer": saving.serialize_keras_object(self.merge_layer),
            "activation": saving.serialize_keras_object(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        })
        
        if self.bottleneck is not None:
            config.update({"bottleneck": saving.serialize_keras_object(self.bottleneck)})
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        activation_cfg = config.pop("activation")
        merge_layer_cfg = config.pop("merge_layer")
        kernel_initializer_cfg = config.pop("kernel_initializer")
        bias_initializer_cfg = config.pop("bias_initializer")
        kernel_regularizer_cfg = config.pop("kernel_regularizer")
        bias_regularizer_cfg = config.pop("bias_regularizer")
        kernel_constraints_cfg = config.pop("kernel_constraint")
        bias_constraints_cfg = config.pop("bias_constraint")
        bottleneck_cfg = config.pop("bottleneck", None)

        # now update with deserialized version
        config.update({
            "activation": saving.deserialize_keras_object(activation_cfg),
            "merge_layer": saving.deserialize_keras_object(merge_layer_cfg),
            "kernel_initializer": initializers.deserialize(kernel_initializer_cfg),
            "bias_initializer": initializers.deserialize(bias_initializer_cfg),
            "kernel_regularizer": regularizers.deserialize(kernel_regularizer_cfg),
            "bias_regularizer": regularizers.deserialize(bias_regularizer_cfg),
            "kernel_constraint": constraints.deserialize(kernel_constraints_cfg),
            "bias_constraint": constraints.deserialize(bias_constraints_cfg)
        })
        if bottleneck_cfg:
            config.update({"bottleneck": saving.deserialize_keras_object(bottleneck_cfg)})

        return cls(**config)
    
    def get_build_config(self) -> dict:
        return self.global_build_shapes_dict
    
    def build_from_config(self, config):
        for layer in self.layers:
            try:
                layer.build(**config[layer.name])
            except ValueError:
                # layer is already build
                pass
            except KeyError:
                # layer has not input shape, e.g., activation layer like ReLU
                pass

        self.built = True


class BaseUnet(BaseFCN):
    """
    Base class of Unet, cf. [Ronneberger et al.](https://arxiv.org/abs/1505.04597)

    Convolutional autoencoder *with* skip-connections

    Notes
    -----
    Downsampling is realized using a strided convolution as proposed by [Springenberger et al.](https://arxiv.org/abs/1412.6806)

    """

    def __init__(self,
            rank,
            filters=[8, 16, 32],
            kernel_size=5,
            padding="same",
            data_format="channels_last",
            dilation_rate=1, 
            groups=1, 
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
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_skip_connection=True,
            activation=activation,
            use_bias=use_bias,
            bottleneck=bottleneck,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            merge_layer=merge_layer,
            **kwargs
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.pop("use_skip_connection", None)  # This should always be `True` for UNet

        return config
    