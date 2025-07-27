from ...layers.conv.base_conv import BaseConv
from ...layers.fno.base_fno import BaseFNO
from keras import models
from ...ops.helper import _IterableVars


class BaseNeuralOperator(models.Model, _IterableVars):
    """
     	
    https://doi.org/10.48550/arXiv.2010.08895

    """

    def __init__(
        self,
        rank,
        filters,
        modes,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.modes = modes

        # define layers
        self.projection_p = BaseConv(rank=self.rank)
        self.fno_layer = BaseFNO(rank=self.rank)
        self.projection_q = BaseConv(rank=self.rank)

    def build(self, input_shape):
        if self.built:
            return
        
        for layer in [self.projection_p, self.fno_layer, self.projection_q]:
            layer.build(input_shape=input_shape)
            input_shape = layer.compute_output_shape(input_shape=input_shape)

        self.built = True

    def call(self, inputs):
        x = self.projection_p(inputs)
        x = self.fno(x)
        x = self.projection_q(x)

        return x
    