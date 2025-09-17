from keras import layers
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import saving
from ...layers.wrapper import Residual
from importlib import import_module


class BaseFNO(layers.Layer):
    """
    Base FNO layer, cf. https://arxiv.org/abs/2010.08895

    This layer has two paths:

    The first path applies a spectral convolution to the input, which
    (1) transforms the layer into Fourier space via discrete Fourier transform for real-valued data (rfft),
    (2) truncates the Fourier modes to the `modes` lowest modes,
    (3) applies the weights (separated into real- and imaginary weights),
    (4) pads the truncated signal in Fourier space back to its initial shape,
    (5) applies inverse discrete Fourier transform for real-valued data (irfft), and
    (6) applies the bias in physical space (if `use_bias=True`)

    The second path applies a 1x1 bypass convolution

    Both path are merged with `merge_layer` and passed through an activation function.    

    Parameters
    ----------
    rank : int {1, 2, 3}
        Rank of the subclassed FNO layer.
    filters : int
        Number of filters.
    modes : int
        Number of modes after truncation in Fourier space.
    activation : str | keras.activations.Activation | keras.layers.Layer, optional
        Activation function.
        Defaults to `"gelu"`.
    merge_layer : str | keras.layers.Layer, optional {`"concatenate"`, `"average"`, `"maximum"`, `"minimum"`, `"add"`, `"subtract"`, `"multiply"`, `"dot"`}
        Merge operation in FNO layers to combine the result from the spectral convolution with the result from the bypass convolution.
        Defaults to `"add"`.
    data_format : str, optional {`"channels_first"`, `"channels_last"`}
        Format of the input data.
        Defaults to `None`.
    use_bias : bool, optional
        If `True`, a bias term is added to the physical space after the spectral convolution.
        Defaults to `True`.
    kernel_initializer : str | keras.initializer.Initializer | tuple, optional
        Initializer for real- and imaginary weights.
        By default, the real weights are initialized using `"glorot_normal"`,
        and the imaginary weights are initialized using `"random"` with a low standard deviation,
        which is effectively white Gaussian noise.
        Defaults to `("glorot_normal", initializers.RandomNormal(stddev=1e-3))`.
    bias_initializer : str | keras.initializer.Initializer, optional
        Initializer for the bias.
        Defaults to `"zeros"`.
    kernel_regularizer : str | keras.regularizers.Regularizer, optional
        Kernel regularizer.
        Defaults to `None`.
    bias_regularizer : str | keras.regularizers.Regularizer, optional
        Bias regularizer.
        Defaults to `None`.
    kernel_constraint : str | keras.constraints.Constraint, optional
        Kernel constraint.
        Defaults to `None`.
    bias_constraint : str | keras.constraints.Constraint, optional
        Bias constraint.
        Defaults to `None`.
    name : str, optional
        Name of the model.
        If `None`, `name` is automatically inherited from the class name `"SpectralConv1D"`.
        Defaults to `None`.

    Notes
    -----
    For implementataion simplicity, the Fourier operations are always performed in `"channels_first"` data format.
    The layer therefore applies a transpose operation if `data_format="channels_last"`.
    
    """

    def __init__(
        self,
        rank,
        filters,
        modes,
        activation="gelu",
        merge_layer="add",
        data_format=None,
        use_bias=True,
        kernel_initializer=("glorot_normal", initializers.RandomNormal(stddev=1e-3)),
        bias_initializer="zeros",
        kernel_constraint=None,
        bias_constraint=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.modes = modes
        self.merge_layer = merge_layer
        self.data_format = data_format
        self.use_bias = use_bias

        if isinstance(kernel_initializer, (list, tuple)):
            self.real_kernel_initializer, self.imag_kernel_initializer = kernel_initializer
        else:
            self.real_kernel_initializer = self.imag_kernel_initializer = kernel_initializer

        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.forward = Residual(
            layer=getattr(import_module(name=".spectral_conv", package=__package__), f"SpectralConv{self.rank}D")(
                filters=self.filters,
                modes=self.modes,
                data_format=self.data_format,
                use_bias=self.use_bias,
                kernel_initializer=(
                    self.real_kernel_initializer,
                    self.imag_kernel_initializer
                ),
                bias_initializer=self.bias_initializer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="spectral_conv",
                **kwargs
            ),
            merge_layer=self.merge_layer,
            residual_layer=getattr(import_module(name="...layers.conv", package=__package__), f"Conv{self.rank}D")(
                filters=self.filters,
                kernel_size=1,
                data_format=self.data_format,
                use_bias=self.use_bias,
                kernel_initializer=self.real_kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="bypass_conv"
            )
        )
        
        self.activation = activations.get(activation)

    def build(self, input_shape):
        if self.built:
            return
        
        self.forward.build(input_shape=input_shape)

        # inherit input spec from `spectral_conv` layer
        self.input_spec = self.forward.layer.input_spec
        
        self.built = True

    def call(self, inputs):
        x = self.forward(inputs)
        x = self.activation(x)

        return x
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape of `BaseNeuralOperator`

        Parameters
        ----------
        input_shape : tuple
            Input shape.

        Returns
        -------
        output_shape : tuple
            Output shape.

        """
        
        return self.forward.compute_output_shape(input_shape=input_shape)        
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "modes": self.modes,
            "activation": saving.serialize_keras_object(self.activation),
            "merge_layer": saving.serialize_keras_object(self.merge_layer),
            "data_format": self.data_format,
            "use_bias": self.use_bias,
            "real_kernel_initializer": initializers.serialize(self.real_kernel_initializer),
            "imag_kernel_initializer": initializers.serialize(self.imag_kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer)
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        activation_cfg = config.pop("activation")
        merge_layer_cfg = config.pop("merge_layer")
        real_kernel_initializer_cfg = config.pop("real_kernel_initializer")
        imag_kernel_initializer_cfg = config.pop("imag_kernel_initializer")
        bias_initializer_cfg = config.pop("bias_initializer")
        kernel_constraint_cfg = config.pop("kernel_constraint")
        bias_constraint_cfg = config.pop("bias_constraint")
        kernel_regularizer_cfg = config.pop("kernel_regularizer")
        bias_regularizer_cfg = config.pop("bias_regularizer")

        config.update({
            "activation": saving.deserialize_keras_object(activation_cfg),
            "merge_layer": saving.deserialize_keras_object(merge_layer_cfg),
            "kernel_initializer": (
                initializers.deserialize(real_kernel_initializer_cfg), 
                initializers.deserialize(imag_kernel_initializer_cfg)
            ),
            "bias_initializer": initializers.deserialize(bias_initializer_cfg),
            "kernel_constraint": constraints.deserialize(kernel_constraint_cfg),
            "bias_constraint": constraints.deserialize(bias_constraint_cfg),
            "kernel_regularizer": regularizers.deserialize(kernel_regularizer_cfg),
            "bias_regularizer": regularizers.deserialize(bias_regularizer_cfg)
        })

        return cls(**config)
    