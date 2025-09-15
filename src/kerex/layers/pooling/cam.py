from keras import saving
from .base_cam import BaseChannelAttentionModule


@saving.register_keras_serializable(package="Kerex.Layers.Pooling", name="ChannelAttentionModule1D")
class ChannelAttentionModule1D(BaseChannelAttentionModule):
    """
    Convolutional Channel Attention Module (CAM)

    Parameters
    ----------
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

    def __init__(
            self, 
            reduction_ratio=8, 
            activation="sigmoid", 
            data_format="channels_last", 
            keepdims=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=1,
            reduction_ratio=reduction_ratio,
            activation=activation,
            data_format=data_format,
            keepdims=keepdims,
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Pooling", name="ChannelAttentionModule2D")
class ChannelAttentionModule2D(BaseChannelAttentionModule):
    """
    Convolutional Channel Attention Module (CAM)

    Parameters
    ----------
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

    def __init__(
            self, 
            reduction_ratio=8, 
            activation="sigmoid", 
            data_format="channels_last", 
            keepdims=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=2,
            reduction_ratio=reduction_ratio,
            activation=activation,
            data_format=data_format,
            keepdims=keepdims,
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.Pooling", name="ChannelAttentionModule3D")
class ChannelAttentionModule3D(BaseChannelAttentionModule):
    """
    Convolutional Channel Attention Module (CAM)

    Parameters
    ----------
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

    def __init__(
            self, 
            reduction_ratio=8, 
            activation="sigmoid", 
            data_format="channels_last", 
            keepdims=True, 
            name=None, 
            **kwargs
        ):
        super().__init__(
            rank=3,
            reduction_ratio=reduction_ratio,
            activation=activation,
            data_format=data_format,
            keepdims=keepdims,
            name=name, 
            **kwargs
        )
