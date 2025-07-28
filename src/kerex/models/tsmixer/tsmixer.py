from keras import layers
from keras import models
from keras import saving
from ...blocks.tsmixer import TSMixerBlock
from ...ops.helper import _IterableVars


@saving.register_keras_serializable(package="KerasAddon.Models", name="TSMixer")
class TSMixer(models.Model, _IterableVars):
    """
    TSMixer model, cf. [Chen et al.](https://doi.org/10.48550/arXiv.2303.06053)

    Parameters
    ----------
    sequence_length : int
        Length of the predicted output sequence.
    num_hidden : int | list | tuple, optional
        Number of units in dense layer.
        Determines the number of TSMixer residual blocks (1 layer if `num_hidden` of type `int` or `None`).
        If `None`, the feature space is expanded by factor of 4 for the respective layer.
        Defaults to `None`.
    norm : str | list | tuple, optional {`"LN"`, `"BN"`}
        Normalization type, can be layer normalization (`"LN"`) or batch normalization (`"BN"`).
        Defaults to `"LN"`.
    activation : str | keras.activations.Activation | keras.layers.Layer, optional
        Activation for dense layers.
        Can be either a `str`, a `keras.activations.Activation`, or a `keras.layers.Layer`.
        Defaults to `"relu"`.
    dropout_rate : float | list | tuple, optional
        Dropout rate.
        Defaults to 0.1.
    name : str, optional
        Name of the layer.
        If `None`, `name` is automatically inherited from the class name `"TSMixerBlock"`.
        Defaults to `None`.
    **kwargs : Additional keyword arguments for the `keras.layers.Layer` super-class.

    Notes
    -----
    Adapted for Keras3 from https://github.com/google-research/google-research/tree/master/tsmixer

    Examples
    --------
    For `None` arguments in `num_hidden`, the respective projection dimension depends on the number of features in the input.
    Explicit arguments of type `int` result in the respective projection dimension.
    The following code creates a TSMixer model with three residual blocks.
    >>> from keras import ops
    >>> x = ops.ones((1, 256, 32))  # time series of length 256 with 32 features
    >>> model = TSMixer(sequence_length=128, num_hidden=[None, 32, 16])
    >>> model.build(input_shape=x.shape)
    >>> y = model(x)
    >>> y.shape
    TensorShape([1, 128, 32])

    """

    def __init__(
        self,
        sequence_length,
        num_hidden=None,
        norm="LN",
        activation="relu",
        dropout_rate=0.1,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.sequence_length = sequence_length

        # residual block arguments
        self.set_vars(num_hidden=num_hidden, norm=norm, dropout_rate=dropout_rate)
        self.activation = activation

        self.residual_blocks = models.Sequential([
            TSMixerBlock(
                num_hidden=n,
                norm=nrm,
                activation=self.activation,
                dropout_rate=d
            ) for n, nrm, d in zip(
                self.num_hidden,
                self.norm,
                self.dropout_rate
            )
        ])

        self.output_section = models.Sequential([
            layers.Permute(dims=(2, 1), name="transpose_1"),
            layers.Dense(units=self.sequence_length, name="dense"),
            layers.Permute(dims=(2, 1), name="transpose_2")
        ])

    def build(self, input_shape):
        if self.built:
            return
        
        self.residual_blocks.build(input_shape=input_shape)
        forward_shape = self.residual_blocks.compute_output_shape(input_shape=input_shape)

        self.output_section.build(input_shape=forward_shape)

        self.built = True

    def call(self, inputs):
        x = self.residual_blocks(inputs)
        x = self.output_section(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "num_hidden": self.num_hidden,
            "norm": self.norm,
            "activation": saving.serialize_keras_object(self.activation),
            "dropout_rate": self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        activation_cfg = config.pop("activation")
        config.update({"activation": saving.deserialize_keras_object(activation_cfg)})

        return cls(**config)
    