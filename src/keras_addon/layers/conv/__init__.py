try:
    import keras
except ImportError as e:
    raise ImportError(
        "The keras package is not installed. It can be installed using 'pip install keras'."
    ) from e

from packaging import version

MIN_KERAS_VERSION = "3.10.0"

if version.parse(keras.__version__) >= MIN_KERAS_VERSION:
    from keras import layers

    Conv1D = layers.Conv1D
    Conv2D = layers.Conv2D
    Conv3D = layers.Conv3D

    Conv1DTranspose = layers.Conv1DTranspose
    Conv2DTranspose = layers.Conv2DTranspose
    Conv3DTranspose = layers.Conv3DTranspose
else:
    """ 
    Keras versions < 3.10.0 had faulty serialization of Convolutional layers,
    c.f. https://github.com/keras-team/keras/issues/21088.

    The following implementation fixes that for Keras<=3.10.0

    """

    from .conv1d import Conv1D, Conv1DTranspose
    from .conv2d import Conv2D, Conv2DTranspose
    from .conv3d import Conv3D, Conv3DTranspose
