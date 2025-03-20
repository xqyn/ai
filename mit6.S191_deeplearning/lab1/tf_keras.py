import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np

# Training data
x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y_train = np.array([0, 1, 0, 1])

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Test data
x_test = np.array([2, 3, 4]).reshape(1, -1)  # Shape: (1, 3)

# Predict
predictions = model.predict(x_test)
print("Predicted probability:", predictions)

# Convert to class label
pred_class = (predictions > 0.5).astype(int)
print("Predicted class:", pred_class)
model.summary()

# --------------------------------------------------
# visualize model
plot_model(model, to_file='tf_model_plot.png', show_shapes=True, show_layer_names=True)

plot_model(
    model,
    to_file='model_plot.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top-to-bottom layout (default); use 'LR' for left-to-right
    expand_nested=True,  # If you have nested models
    dpi=96  # Resolution of the image
)

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

# visualize model

import matplotlib.pyplot as plt
import numpy as np

# Define neuron positions
input_neurons = 5
hidden_neurons = 10
output_neurons = 1

# Positions for each layer (x, y coordinates)
input_pos = [(0, i) for i in np.linspace(0, 4, input_neurons)]
hidden_pos = [(2, i) for i in np.linspace(0, 9, hidden_neurons)]
output_pos = [(4, 0)]  # Single output neuron

# Plot neurons as circles
fig, ax = plt.subplots(figsize=(8, 6))

# Input layer
for x, y in input_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='blue'))
    # Connect to hidden layer
    for hx, hy in hidden_pos:
        ax.plot([x, hx], [y, hy], 'gray', linestyle='-', alpha=0.3)

# Hidden layer
for x, y in hidden_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='green'))
    # Connect to output layer
    for ox, oy in output_pos:
        ax.plot([x, ox], [y, oy], 'gray', linestyle='-', alpha=0.3)

# Output layer
for x, y in output_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='red'))

# Labels
ax.text(-0.5, 4.5, 'Input Layer (5)', fontsize=12, color='blue')
ax.text(1.5, 9.5, 'Hidden Layer (10)', fontsize=12, color='green')
ax.text(3.5, 0.5, 'Output Layer (1)', fontsize=12, color='red')

# Set limits and remove axes
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')
ax.axis('off')

plt.title("Neural Network Visualization")
plt.savefig('neural_network.png')
plt.show()

# --------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Define neuron counts
input_neurons = 5
hidden_neurons = 10
output_neurons = 1

# Neuron positions (x for layer, y for neuron placement)
input_pos = [(0, i) for i in np.linspace(0, 4, input_neurons)]
hidden_pos = [(2, i) for i in np.linspace(0, 9, hidden_neurons)]
output_pos = [(4, 4.5)]  # Center the single output neuron

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

# Draw input layer
for x, y in input_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='blue'))
    for hx, hy in hidden_pos:
        ax.plot([x, hx], [y, hy], 'gray', linestyle='-', alpha=0.2)

# Draw hidden layer
for x, y in hidden_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='green'))
    for ox, oy in output_pos:
        ax.plot([x, ox], [y, oy], 'gray', linestyle='-', alpha=0.2)

# Draw output layer
for x, y in output_pos:
    ax.add_patch(plt.Circle((x, y), 0.2, fill=True, color='red'))

# Add layer labels
ax.text(-0.5, 4.5, 'Input (5)', fontsize=12, color='blue')
ax.text(1.5, 9.5, 'Hidden (10)', fontsize=12, color='green')
ax.text(3.5, 5, 'Output (1)', fontsize=12, color='red')

# Customize plot
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')
ax.axis('off')
plt.title("Neural Network: 5 -> 10 -> 1")
plt.savefig('neural_network.png')
plt.show()