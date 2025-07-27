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
        name=None,
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            modes=modes,
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
        name=None,
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            modes=modes,
            name=name,
            **kwargs
        )
        