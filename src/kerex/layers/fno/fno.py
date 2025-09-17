from keras import saving
from keras import initializers
from .base_fno import BaseFNO


@saving.register_keras_serializable(package="Kerex.Layers.FNO", name="FNO1D")
class FNO1D(BaseFNO):
    """
    FNO1D layer, cf. https://arxiv.org/abs/2010.08895

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
        If `None`, `name` is automatically inherited from the class name `"FNO1D"`.
        Defaults to `None`.

    Notes
    -----
    For implementataion simplicity, the Fourier operations are always performed in `"channels_first"` data format.
    The layer therefore applies a transpose operation if `data_format="channels_last"`.
    
    """

    def __init__(
        self, 
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
        super().__init__(
            rank=1, 
            filters=filters, 
            modes=modes, 
            activation=activation, 
            merge_layer=merge_layer,  
            data_format=data_format, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.FNO", name="FNO2D")
class FNO2D(BaseFNO):
    """
    FNO2D layer, cf. https://arxiv.org/abs/2010.08895

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
        If `None`, `name` is automatically inherited from the class name `"FNO2D"`.
        Defaults to `None`.

    Notes
    -----
    For implementataion simplicity, the Fourier operations are always performed in `"channels_first"` data format.
    The layer therefore applies a transpose operation if `data_format="channels_last"`.
    
    """

    def __init__(
        self, 
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
        super().__init__(
            rank=2, 
            filters=filters, 
            modes=modes, 
            activation=activation, 
            merge_layer=merge_layer,  
            data_format=data_format, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            name=name, 
            **kwargs
        )


@saving.register_keras_serializable(package="Kerex.Layers.FNO", name="FNO3D")
class FNO3D(BaseFNO):
    """
    FNO3D layer, cf. https://arxiv.org/abs/2010.08895

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
        If `None`, `name` is automatically inherited from the class name `"FNO3D"`.
        Defaults to `None`.

    Notes
    -----
    For implementataion simplicity, the Fourier operations are always performed in `"channels_first"` data format.
    The layer therefore applies a transpose operation if `data_format="channels_last"`.
    
    """

    def __init__(
        self, 
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
        super().__init__(
            rank=3, 
            filters=filters, 
            modes=modes, 
            activation=activation, 
            merge_layer=merge_layer,  
            data_format=data_format, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=bias_constraint, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            name=name, 
            **kwargs
        )
        