from keras import models
from keras import saving
from ...blocks.tsmixer import TSMixerBlock
from ...ops.helper import _IterableVars


@saving.register_keras_serializable(package="KerasAddon.Models", name="TSMixer")
class TSMixer(models.Model, _IterableVars):
    def __init__(
        self,
        num_hidden=None,
        norm="LN",
        activation="relu",
        dropout_rate=0.1,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.set_vars(num_hidden=num_hidden, norm=norm, dropout_rate=dropout_rate)
        self.activation = activation

        self.forward = models.Sequential([
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

    def build(self, input_shape):
        if self.built:
            return
        
        self.forward.build(input_shape=input_shape)

        self.built = True

    def call(self, inputs):
        return self.forward(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
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
    