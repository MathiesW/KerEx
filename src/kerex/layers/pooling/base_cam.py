from keras import layers
from keras import Sequential
from keras import ops
from keras.src import activations
from keras.src.backend import standardize_data_format
from ...ops import get_layer


class BaseChannelAttentionModule(layers.Layer):
    """
    Convolutional Channel Attention Module (CAM)

    Parameters
    ----------
    rank : int
        Rank of the convolutions.
    reduction_ratio : int, optional
        Size of the bottleneck of the shared MLP in relation to the input size.
        Defaults to 8.
    activation : str, optional
        Activation function for the shared MLP.
        Defaults to `"sigmoid"`.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Data format of the convolutions.
        Can be either `"channels_first"` or `"channels_last"`.
        Defaults to `"channels_last"`.
    keepdims : bool, optional
        If `True`, the dimensions are preserved (as 1) throughout the pooling process.
        Select `False` to get a global pooling layer that automatically weighs the importance of channels.
        Defaults to `True`.
    name : str, optional
        Name of the layer.
        Defaults to `None`.
    **kwargs : Additional keyword arguments for the `Layer` super-class.

    Notes
    -----
    This is not a wrapper but a CAM layer.
    May be used on it's own, but here it serves as a building block for the 
    [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521).
    
    More information on the CAM can be found here: https://www.digitalocean.com/community/tutorials/attention-mechanisms-in-computer-vision-cbam#channel-attention-module-cam
    
    """

    def __init__(self, rank, reduction_ratio=8, activation="sigmoid", data_format="channels_last", keepdims=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rank = rank
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.data_format = standardize_data_format(data_format)
        self.keepdims = keepdims
        self.channel_axis = -1 if self.data_format == "channels_last" else 1

        self.avg_pooling = get_layer(f"GlobalAveragePooling{self.rank}D", data_format=self.data_format, keepdims=self.keepdims)
        self.max_pooling = get_layer(f"GlobalMaxPooling{self.rank}D", data_format=self.data_format, keepdims=self.keepdims)
        self.merge_layer = layers.Concatenate(axis=self.channel_axis)
        self.activation_fn = activations.get(self.activation)

    def build(self, input_shape):
        """
        Build method of `ChannelAttentionModule`
        
        Builds all sub-modules and sets `self.built=True`.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        """

        if self.built:
            return

        # define mlp
        input_shape = list(input_shape)
        num_ch = input_shape[self.channel_axis]
        self.mlp = Sequential([
            layers.Dense(units=num_ch // self.reduction_ratio, activation="relu"),
            layers.Dense(units=num_ch)
        ])

        # forward pass with updated input shapes
        self.max_pooling.build(input_shape=input_shape)
        self.avg_pooling.build(input_shape=input_shape)

        maxpool_output_shape = self.max_pooling.compute_output_shape(input_shape=input_shape)
        avgpool_output_shape = self.avg_pooling.compute_output_shape(input_shape=input_shape)

        self.merge_layer.build(input_shape=(maxpool_output_shape, avgpool_output_shape))

        merge_output_shape = self.merge_layer.compute_output_shape(input_shape=(maxpool_output_shape, avgpool_output_shape))

        self.mlp.build(input_shape=merge_output_shape)

        self.built = True

    def call(self, inputs):
        """
        Call method of `ChannelAttentionModule`

        Parameters
        ----------
        inputs : KerasTensor
            Input tensor for `ChannelAttentionModule`.

        Returns
        -------
        outputs : KerasTensor
            Refined features.

        """

        max_pool = self.max_pooling(inputs)  # shape (batch, 1, 1, ch)
        avg_pool = self.avg_pooling(inputs)  # shape (batch, 1, 1, ch)

        # concatenate
        x = ops.concatenate([max_pool, avg_pool], axis=self.channel_axis)  # shape = (batch, 1, 1, 2*ch)

        # apply mlp
        x = self.mlp(x)
        x = self.activation_fn(x)

        return x
    
    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape

        Parameters
        ----------
        input_shape : tuple
            Shape of the input.

        Returns
        -------
        output_shape : tuple
            Shape of the output of `ChannelAttentionModule`.

        """

        maxpool_output_shape = self.max_pooling.compute_output_shape(input_shape=input_shape)
        avgpool_output_shape = self.avg_pooling.compute_output_shape(input_shape=input_shape)
        merge_output_shape = self.merge_layer.compute_output_shape(input_shape=(maxpool_output_shape, avgpool_output_shape))
        mlp_output_shape = self.mlp.compute_output_shape(input_shape=merge_output_shape)

        return tuple(mlp_output_shape)

    def get_config(self):
        """
        Get config method.
        Required for serialization.

        Returns
        -------
        config : dict
            Dictionary with the configuration of `ChannelAttentionModule`

        """
        
        config: dict = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "activation": self.activation,
            "data_format": self.data_format,
            "keepdims": self.keepdims
        })
        return config