from keras import layers, Layer
from keras import InputSpec
from keras.src.backend import standardize_data_format
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import ops


class BaseAttentionPooling(Layer):
    """
    Base layer for attention based global pooling
    Each query represents a kind of feature or pattern in the input,
    which the input is attended to.

    Parameters
    ----------
    rank : int {1, 2, 3}
        Rank of the subclassed layer
    num_queries : int,
        Number of learned queries.
    num_heads : int, optional
        Number of heads in the multihead attention layer.
        Defaults to `4`.
    flatten : bool, optional
        If `True`, the output is flattened over the number of input channels and the number of queries.
        Defaults to `True`.
    kernel_initializer : str | keras.initializer.Initializer, optional
        Initializer for real- and imaginary weights.
        Defaults to `"glorot_uniform"`.
    kernel_regularizer : str | keras.regularizers.Regularizer, optional
        Kernel regularizer.
        Defaults to `None`.
    kernel_constraint : str | keras.constraints.Constraint, optional
        Kernel constraint.
        Defaults to `None`.
    name : str, optional
        Name of the model.
        If `None`, `name` is automatically inherited from the class name `"BaseSpectralConv"`.
        Defaults to `None`.

    Notes
    -----
    If `flatten=True`, this layer is basically a drop-in replacement for any global pooling layer,
    The output channels depend on the input channels and the number of queries:
    `out_channels = in_channels * num_queries`.

    """
    
    def __init__(
            self, 
            rank,
            num_queries, 
            num_heads=4,
            flatten=True,
            data_format=None,
            kernel_initializer="glorot_uniform",
            kernel_constraint=None,
            kernel_regularizer=None,
            name=None, 
            **kwargs
        ):
        super().__init__(
            name=name,
            **kwargs
        )
        self.rank = rank
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        self.flatten = flatten

        self.data_format = standardize_data_format(data_format)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        if self.flatten:
            self.flattening_layer = layers.Flatten(data_format=self.data_format)

    def build(self, input_shape):
        if self.built:
            return
        
        # build super class
        super().build(input_shape)

        if self.data_format == "channels_last":
            channel_axis = -1
            d_emb = input_shape[-1]
            target_shape = [-1, d_emb]

        else:
            channel_axis = 1
            d_emb = input_shape[1]
            target_shape = [d_emb, -1]

        # reshape layer, flattens input if necessary
        self.reshape = layers.Reshape(target_shape=target_shape)

        # define input spec
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: d_emb}
        )
        
        # add learnable query vector
        self.query = self.add_weight(
            name="learnable_queries",
            shape=(1, self.num_queries, d_emb),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )

        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=d_emb
        )

    def call(self, inputs):
        flat_inputs = self.reshape(inputs)

        # tile queries across batch
        b = ops.shape(inputs)[0]
        q = ops.tile(self.query, [b, 1, 1])

        pooled = self.mha(query=q, value=flat_inputs)

        if self.flatten:
            pooled = self.flattening_layer(pooled)

        return pooled
    
    def compute_output_shape(self, input_shape):
        b = input_shape[0]
        d_emb = input_shape[-1 if self.data_format == "channels_last" else 1]

        flat_input_shape = self.reshape.compute_output_shape(input_shape)

        pooled_shape = self.mha.compute_output_shape(
            query_shape=tuple([b, self.num_queries, d_emb]),
            value_shape=flat_input_shape
        )

        if self.flatten:
            pooled_shape = self.flattening_layer.compute_output_shape(pooled_shape)

        return pooled_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_queries": self.num_queries,
            "num_heads": self.num_heads,
            "flatten": self.flatten,
            "data_format": self.data_format,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer)
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        kernel_initializer_cfg = config.pop("kernel_initializer")
        kernel_constraint_cfg = config.pop("kernel_constraint")
        kernel_regularizer_cfg = config.pop("kernel_regularizer")

        config.update({
            "kernel_initializer": initializers.deserialize(kernel_initializer_cfg),
            "kernel_constraint": constraints.deserialize(kernel_constraint_cfg),
            "kernel_regularizer": regularizers.deserialize(kernel_regularizer_cfg)
        })

        return cls(**config)
    