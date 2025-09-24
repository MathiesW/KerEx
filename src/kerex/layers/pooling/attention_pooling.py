from .base_attention_pooling import BaseAttentionPooling
from keras import saving


@saving.register_keras_serializable(package="Kerex.layers.pooling", name="AttentionPooling1D")
class AttentionPooling1D(BaseAttentionPooling):
    """
    1D attention based global pooling
    Each query represents a kind of feature or pattern in the input,
    which the input is attended to.

    Parameters
    ----------
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
            rank=1, 
            num_queries=num_queries, 
            num_heads=num_heads, 
            flatten=flatten, 
            data_format=data_format, 
            kernel_initializer=kernel_initializer, 
            kernel_constraint=kernel_constraint, 
            kernel_regularizer=kernel_regularizer, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.layers.pooling", name="AttentionPooling2D")
class AttentionPooling2D(BaseAttentionPooling):
    """
    2D attention based global pooling
    Each query represents a kind of feature or pattern in the input,
    which the input is attended to.

    Parameters
    ----------
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
            rank=2, 
            num_queries=num_queries, 
            num_heads=num_heads, 
            flatten=flatten, 
            data_format=data_format, 
            kernel_initializer=kernel_initializer, 
            kernel_constraint=kernel_constraint, 
            kernel_regularizer=kernel_regularizer, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.layers.pooling", name="AttentionPooling3D")
class AttentionPooling3D(BaseAttentionPooling):
    """
    3D attention based global pooling
    Each query represents a kind of feature or pattern in the input,
    which the input is attended to.

    Parameters
    ----------
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
            rank=3, 
            num_queries=num_queries, 
            num_heads=num_heads, 
            flatten=flatten, 
            data_format=data_format, 
            kernel_initializer=kernel_initializer, 
            kernel_constraint=kernel_constraint, 
            kernel_regularizer=kernel_regularizer, 
            name=name, 
            **kwargs
        )
