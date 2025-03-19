"""
19 march 2025 - XQ - Leiden UMC
re-creating the builing block of neuron network myself
"""

#!pip install mitdeeplearning --quiet

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import mitdeeplearning as mdl

# --------------------------------------------------


# super(OurDenseLayer, self).__init__() calls the parent class’s __init__ method for the current instance (self).
# It ensures the parent class’s initialization runs, setting up inherited features.
# In your case, it prepares OurDenseLayer to work as a proper subclass of its parent (e.g., a neural network layer class).
# In Python 3, you could simplify it to super().__init__().

class DenseLayers(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(DenseLayers, self).__init__()
        self.n_output_nodes = n_output_nodes
    
    def build(self, input_shape):
        d = int(input_shape[-1])
        # calling weight
        self.W = self.add_weight(name='weight', shape=[d, self.n_output_nodes])
        self.b = self.add_weight(name='bias', shape=[1, self.n_output_nodes])
    
    def call(self, x):
        # computing function node
        z = tf.matmul(x, self.W) + self.b
        # activation node
        y = tf.sigmoid(z)
        return y


# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.keras.utils.set_random_seed(1)
layer = DenseLayers(3)
layer.build((1,2))
x_input = tf.constant([[1,2.]], shape=(1,2))
y = layer.call(x_input)

        
        