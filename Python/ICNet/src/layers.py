# coding: utf-8
"""
Custom layers that define the DNN architecture.

Fotios Drakopoulos, UCL, June 2024
"""

__author__ = 'fotisdr'
__contributors__ = 'afragner'


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Layer, Reshape
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

import numpy as np
import math

from activations import *

@register_keras_serializable(package=__name__)
class ConvEncoderBlock(Layer):
    """
    A simple 1D convolutional encoder block
    """
    def __init__(self, channels, kernel_size, padding, activation, 
                 strides, activity_regularizer=None, kernel_regularizer=None, 
                 kernel_initializer='glorot_uniform', layer_normalization=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.channels = channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,)
        else:
            self.kernel_size = kernel_size

        self.padding = padding
        self.activation = activation
        self.strides = strides
        self.activity_regularizer = activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.layer_normalization = layer_normalization # unused

        self.conv = Conv1D(
            channels,
            kernel_size,
            padding=padding,
            strides=strides,
            activity_regularizer = activity_regularizer,
            kernel_regularizer = kernel_regularizer,
            kernel_initializer = kernel_initializer,
            name=f"{self.name}_conv",
        )
        self.out = get_activation(activation, name=f"{self.name}_out")

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.out(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "activation": self.activation,
                "strides": self.strides,
                "activity_regularizer": self.activity_regularizer,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_initializer": self.kernel_initializer,
                "layer_normalization": self.layer_normalization,
            }
        )
        return config
    
@register_keras_serializable(package=__name__)
class SincNetBlock(Layer):
    """
    A 1D convolutional SincNet encoder block
    """

    def __init__(self, channels, kernel_size, padding, activation, 
                 strides, fs=24414.0625, **kwargs):
        super().__init__(**kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.strides = strides
        self.fs = fs

        self.conv = SincConv1D(N_filt=channels,
                                Filt_dim=kernel_size + 1,
                                fs=fs,
                                stride=strides,
                                padding=padding,
                                name=f"{self.name}_conv",
                              )
        self.out = get_activation(activation, name=f"{self.name}_out")

    def call(self, inputs):
        outputs = self.conv(inputs)
        return self.out(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "activation": self.activation,
                "strides": self.strides,
                "fs": self.fs,
            }
        )
        return config
    
@register_keras_serializable(package=__name__)
class CrossEntropyBlock(Layer):
    """
    A block that maps an output to different classes (probabilities),
    followed by a softmax activation across all classes.
    """

    def __init__(self, output_channels, parameters=5, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters # number of classes
        self.output_channels = output_channels

    def call(self, inputs):
        newshape = inputs.shape[1:-1] + int(inputs.shape[-1]/self.parameters) + (self.parameters,)
        # Convert the None dimensions to -1
        newshape = tuple([-1 if dim == None else dim for dim in newshape])
        # Reshape to add a new dimension at the end
        outputs = Reshape(newshape, name=f"{self.name}_reshape")(inputs)
        # Apply a softmax activation to bound p between 0 and 1 (and the sum to 1)
        outputs = get_activation('softmax', name=f"{self.name}_p")(outputs)
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "parameters": self.parameters,
                "output_channels": self.output_channels,
            }
        )
        return config

@register_keras_serializable(package=__name__)
class SincConv1D(Layer):
    """
    SincNet convolutional layer, adapted from:
    https://github.com/grausof/keras-sincnet/blob/master/sincnet.py
    """
    def __init__(
            self,
            N_filt,
            Filt_dim,
            fs,
            stride=128,
            padding="SAME",
            low_freq_mel = None,
            high_freq_mel = None,
            **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.stride = stride
        self.padding = padding
        # Define the frequency range here
        # Different initial values can be used for the layer 
        if low_freq_mel == None and high_freq_mel == None:
            self.low_freq_mel = 80
            self.high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700)) # Convert Hz to Mel
        else:
            self.low_freq_mel = (2595 * np.log10(1 + low_freq_mel / 700)) # Convert Hz to Mel
            self.high_freq_mel = (2595 * np.log10(1 + high_freq_mel / 700)) # Convert Hz to Mel
        
        super(SincConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name='filt_b1',
            shape=(self.N_filt, 1),
            initializer='uniform',
            trainable=True)
        self.filt_band = self.add_weight(
            name='filt_band',
            shape=(self.N_filt, 1),
            initializer='uniform',
            trainable=True)

        # Mel Initialization of the filterbanks
        low_freq_mel = self.low_freq_mel
        high_freq_mel = self.high_freq_mel
        
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.B1 = np.expand_dims(b1, axis=-1)
        self.B2 = np.expand_dims(b2, axis=-1)
        self.freq_scale = self.fs * 1.0

        t_right = tf.constant(tf.linspace(1.0, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2)) / self.fs, tf.float32)
        self.T_Right = tf.tile(tf.expand_dims(t_right, axis=0), (self.N_filt, 1))

        n = tf.linspace(0, self.Filt_dim - 1, self.Filt_dim)
        window = 0.54 - 0.46 * tf.cos(2 * math.pi * n / self.Filt_dim)
        window = tf.cast(window, tf.float32)
        self.Window = tf.tile(tf.expand_dims(window, axis=0), (self.N_filt, 1))


        self.set_weights([self.B1 / self.freq_scale,
                         (self.B2 - self.B1) / self.freq_scale])
        
        super(SincConv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        
        min_freq = 50.0;
        min_band = 50.0;
        
        filt_beg_freq = tf.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (tf.abs(self.filt_band) + min_band / self.freq_scale)

        low_pass1 = 2 * filt_beg_freq * self.sinc(filt_beg_freq * self.freq_scale)
        low_pass2 = 2 * filt_end_freq * self.sinc(filt_end_freq * self.freq_scale)
        band_pass = (low_pass2 - low_pass1)
        band_pass = band_pass / tf.reduce_max(band_pass, axis=1, keepdims=True)
        windowed_band_pass = band_pass * self.Window
        
        filters = tf.transpose(windowed_band_pass)
        filters = tf.reshape(filters, (self.Filt_dim, 1, self.N_filt))

        # Do the convolution.
        out = K.conv1d(
            x,
            kernel=filters,
            strides=self.stride,
            padding=self.padding
        )

        return out

    def sinc(self, band):
        y_right = tf.sin(2 * math.pi * band * self.T_Right) / (2 * math.pi * band * self.T_Right)
        y_left = tf.reverse(y_right, [1])
        y = tf.concat([y_left, tf.ones((self.N_filt, 1)), y_right], axis=1)
        return y

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding=self.padding.lower(),
            stride=self.stride,
            dilation=1)
        return (input_shape[0],) + (new_size,) + (self.N_filt,)
