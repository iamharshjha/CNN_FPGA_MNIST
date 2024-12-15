import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
n = 100
for i in range(n):

# Select a random image from the dataset, for example, the first image
    input_img = x_train[0]  # Choose an image from the training set

    # Normalize the image to match the range of values in your example (0-1)
    input_img = input_img.astype('float32') / 255.0  # Scaling the values to [0, 1]

    # Reshape to add the channel dimension (28x28x1)
    input_img = np.expand_dims(input_img, axis=-1)  # Adding the channel dimension (1 for grayscale)

    print("Shape of input image:", input_img.shape)