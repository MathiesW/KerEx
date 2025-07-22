from keras import Model
from keras import activations, regularizers, initializers, constraints
from keras import saving
from ...blocks.autoencoder.base_encoder import BaseEncoder
from ...blocks.autoencoder.base_decoder import BaseDecoder
from ...ops import get_layer


# @saving.register_keras_serializable(package="BaseUnet", name="forward")
# def forward(inputs: KerasTensor) -> KerasTensor:
#     return inputs

class BaseFCN(Model):
    def __init__(
            self,
            rank,
            filters=[8, 16, 32],
            kernel_size=[5, 5, 5],
            data_format="channels_last",
            activation="relu",
            use_bias=True,
            bottleneck=None,
            kernel_initializer="HeNormal",
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
            use_skip_connection=False,
            **kwargs
        ):
        
        self.merge_layer = kwargs.pop("merge_layer", "concatenate")
        super().__init__(
            activity_regularizer=activity_regularizer, 
            trainable=trainable, 
            dtype=dtype, 
            autocast=autocast, 
            name=name, 
            **kwargs
        )
        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activity_regularizer = activity_regularizer

        self.use_skip_connection = use_skip_connection

        self.set_encoder_layers()
        self.bottleneck = bottleneck
        self.set_decoder_layers()

        self.global_build_shapes_dict = None
        

    def set_encoder_layers(self):
        self.encoder_layers = [
            BaseEncoder(
                rank=self.rank,
                filters=f,
                kernel_size=k,
                data_format=self.data_format,
                dilation_rate=1,
                groups=1,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                use_skip_connection=self.use_skip_connection,
                activity_regularizer=self.activity_regularizer,
                name=f"Encoder_{i}"
            ) for i, (f, k) in enumerate(zip(self.filters, self.kernel_size))
        ]
    
    def set_decoder_layers(self):
        self.decoder_layers = [
            BaseDecoder(
                rank=self.rank,
                filters=f,
                kernel_size=k,
                data_format=self.data_format,
                dilation_rate=1,
                groups=1,
                activation=self.activation,
                use_bias=self.use_bias,
                merge_layer=get_layer(self.merge_layer), 
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                activity_regularizer=self.activity_regularizer,
                name=f"Decoder_{len(self.filters) - i - 1}"
            ) for i, (f, k) in enumerate(zip(list(reversed(self.filters)), list(reversed(self.kernel_size))))
        ]

    def build(self, input_shape: tuple):
        if self.built:
            return
        
        # cache build shapes
        # self.global_build_shapes_dict = {"input_shapes": [], "skip_shapes": None}
        self.global_build_shapes_dict = {}  # layer.name: None for layer in self.layers}

        # build encoder layers
        output_shapes = []
        for layer in self.encoder_layers:
            # cache input shape
            # self.global_build_shapes_dict["input_shapes"].append(input_shape)
            self.global_build_shapes_dict.update({layer.name: {"input_shape": input_shape}})

            layer.build(input_shape=input_shape)

            # update input shape and append to output_shapes
            input_shape = layer.compute_output_shape(input_shape=input_shape)
            if self.use_skip_connection:
                input_shape, input_shape_skip = input_shape
                output_shapes.append(input_shape_skip)
            else:
                output_shapes.append(None)

        # build bottleneck layers
        if self.bottleneck is not None:
            # cache input shape
            # self.global_build_shapes_dict["input_shapes"].append(input_shape)
            self.global_build_shapes_dict.update({self.bottleneck.name: {"input_shape": input_shape}})

            self.bottleneck.build(input_shape=input_shape)
            input_shape = self.bottleneck.compute_output_shape(input_shape=input_shape)

        # now build decoder layers
        output_shapes.reverse()

        # cache build shapes
        # self.global_build_shapes_dict["skip_shapes"] = output_shapes

        for layer, input_shape_skip in zip(self.decoder_layers, output_shapes):
            # cache input shape
            # self.global_build_shapes_dict["input_shapes"].append(input_shape)
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
        for layer in self.encoder_layers:
            x = layer(x)
            if self.use_skip_connection:
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
    
    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update({
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "activation": saving.serialize_keras_object(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "use_skip_connection": self.use_skip_connection,
            "merge_layer": saving.serialize_keras_object(self.merge_layer)
        })
        
        if self.bottleneck is not None:
            config.update({"bottleneck": saving.serialize_keras_object(self.bottleneck)})
        return config
    
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
    def __init__(self,
            rank,
            filters=[8, 16, 32],
            kernel_size=[5, 5, 5],
            data_format="channels_last",
            activation="relu",
            use_bias=True,
            bottleneck=None,
            merge_layer="concatenate",
            kernel_initializer="HeNormal",
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
            use_skip_connection=True,
            **kwargs
    ):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bottleneck=bottleneck,
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
            use_skip_connection=use_skip_connection,
            merge_layer=merge_layer,
            **kwargs
        )
    