import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np



# n_number_node : number of nodes in the hidden layer
tf.keras.utils.set_random_seed(1)


# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

n_output_nodes = 
input_shape = 

model = tf.keras.Sequential([
])

# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the number of outputs
n_output_nodes = 3

# First define the model
model = Sequential()

''': Define a dense (fully connected) layer to compute z'''
dense_layer = tf.keras.layers.Dense(n_output_nodes, 
                          input_shape=input_shape,
                          use_bias=True)
# Add the dense layer to the model
model.add(dense_layer)
dense_layer = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)
model.add(dense_layer)