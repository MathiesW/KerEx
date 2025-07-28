from keras import saving
from .base_neural_operator import BaseNeuralOperator


@saving.register_keras_serializable(package="KerasAddon.Models.NeuralOperator", name="NeuralOperator1D")
class NeuralOperator1D(BaseNeuralOperator):
    """
     	
    https://doi.org/10.48550/arXiv.2010.08895

    """

    def __init__(
        self,
        filters,
        modes,
        input_projection_dimension=None,
        output_projection_dimension=None,
        data_format="channels_last",
        merge_layer="add",
        activation="relu",
        use_bias=True,
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
            rank=1,
            filters=filters,
            modes=modes,
            input_projection_dimension=input_projection_dimension,
            output_projection_dimension=output_projection_dimension,
            data_format=data_format,
            merge_layer=merge_layer,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )


@saving.register_keras_serializable(package="KerasAddon.Models.NeuralOperator", name="NeuralOperator2D")
class NeuralOperator2D(BaseNeuralOperator):
    """
     	
    https://doi.org/10.48550/arXiv.2010.08895

    """
    
    def __init__(
        self,
        filters,
        modes,
        input_projection_dimension=None,
        output_projection_dimension=None,
        data_format="channels_last",
        merge_layer="add",
        activation="relu",
        use_bias=True,
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
            rank=2,
            filters=filters,
            modes=modes,
            input_projection_dimension=input_projection_dimension,
            output_projection_dimension=output_projection_dimension,
            data_format=data_format,
            merge_layer=merge_layer,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )
        