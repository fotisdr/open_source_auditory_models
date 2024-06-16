# coding: utf-8
"""
Activation layers for the DNN architecture.

Fotios Drakopoulos, UCL, June 2024
"""

__author__ = 'fotisdr'
__contributors__ = 'afragner'


from typing import Optional

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Activation, PReLU, ReLU, LeakyReLU, Layer
from tensorflow.keras.utils import register_keras_serializable

def symmetric_log(x):
    """ Custom symmetric log function
    """
    return tf.math.sign(x) * tf.math.log(tf.math.abs(x) + 1)

@register_keras_serializable(package=__name__)
class get_activation(Layer):
    """
    Get Keras activations by string name (tensorflow layer)

    Parameters
    ----------
    value : str
        string name of activation layer
    name : str, optional
        optional name attribute of returned instance
    """
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)

        self.value = value
        # Define the activation based on the input string
        if value == "prelu":
            rv = PReLU(shared_axes=[1], name=f"{self.name}")
        elif value == "symlog":
            rv = Activation(symmetric_log, name=f"{self.name}")
        elif value == "softmax":
            rv = Activation(activations.softmax, name=f"{self.name}")
        else:
            raise ValueError(f"Activation '{value}' not supported.")
        # Return the activation function
        self.rv = rv

    def call(self, inputs):
        # Compute the activation
        outputs = self.rv(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {"value": self.value,}
        )
        return config
