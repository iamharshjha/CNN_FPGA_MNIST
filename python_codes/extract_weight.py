from tensorflow import keras
import numpy as np
import os
def get_weights(weights , filename):
    os.makedirs("./weights", exist_ok=True)

    # Open the file for writing (it will create the file if it doesn't exist)
    with open(f"./weights/{filename}.txt", "w") as file:
        file.write(str(weights))


model = keras.models.load_model("./trained_model.h5" , compile = True)
layer0_separable_conv2d = model.layers[0]  # Assuming this is a SeparableConv2D layer
layer2_separable_conv2d = model.layers[2]  # Assuming this is also a SeparableConv2D layer
layer4_flatten = model.layers[4]
layer5_dense = model.layers[5]
layer6_dense = model.layers[6]

             # Dense layer

    # Extract weights
layer0_separable_conv2d_weights = layer0_separable_conv2d.get_weights()  # Includes depthwise and pointwise weights
layer2_separable_conv2d_weights = layer2_separable_conv2d.get_weights()
layer4_flatten_weights = layer4_flatten.get_weights()
layer5_dense_weights = layer5_dense.get_weights()
layer6_dense_weights = layer6_dense.get_weights()  # Includes weights and biases
# print(model.layers[7])
#     # Print all weights with no threshold limit
# np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
    # Dump weights (custom function you are using)
get_weights(layer0_separable_conv2d_weights, "layer0_separable_conv2d")
get_weights(layer2_separable_conv2d_weights, "layer2_separable_conv2d")
get_weights(layer4_flatten_weights, filename="layer4_flatten_weights")
get_weights(layer5_dense_weights, filename="layer5_dense_weights")
get_weights(layer6_dense_weights, "layer6_dense")


