"""
This code trains a simple neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.
It then evaluates the model on the test set and prints the test accuracy.
"""

import tensorflow as tf # library for machine learning and deep learning tasks
from tensorflow.keras import layers # Keras layers module from TensorFlow
from tensorflow.keras.datasets import mnist # dataset from the Keras datasets module, handwritten digits

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Loads the MNIST dataset and splits it into training and testing sets.

# Converts the training and test data to the float32 data type and normalizes the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Converts the training and testing labels into one-hot encoded vectors with 10 categories (0-9)
# about one-hot vectors: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Defines the neural network model using Keras Sequential API. It consists of three layers:
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Flattens the input images (28x28 pixels) into a single vector of 784 elements
    layers.Dense(128, activation='relu'), # A fully connected (dense) layer with 128 units and the ReLU activation function
    layers.Dense(10, activation='softmax') # A fully connected (dense) layer with 10 units and the softmax activation function, which outputs the probability distribution over the 10 classes (digits)
])

# Configures the model for training by specifying the optimizer (Adam), loss function (categorical cross-entropy), and evaluation metric (accuracy)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains the model on the training data for 5 epochs, using a batch size of 32 and 20% of the data as validation data
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#Evaluates the model on the testing data and returns the loss and accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")