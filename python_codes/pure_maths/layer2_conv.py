import numpy as np
from layer1_conv import *

path = '../weights/layer2_separable_conv2d.json'
weights_and_biases_2 = load_weights_from_json(path)

depthwise_weights_2 = np.array(weights_and_biases_2[:50]).reshape((2 , 5, 5 ))  # 5x5 kernel for 1 channel
pointwise_weights_2 = np.array(weights_and_biases_2[50:58]).reshape((1, 1, 2, 4))  # 1x1 kernel for 2 output channels
pointwise_biases_2 = np.array(weights_and_biases_2[58:62])  # Biases for the pointwise convolution


input_data = np.random.rand(28,28,1)
#layer1_output = process_layer1(input_data)
layer2_input = process_layer1(input_data)
layer2_output = depthwise_sep_conv(layer2_input, depthwise_weights_2, pointwise_weights_2,
                                       depthwise_kernel_size=5, pointwise_kernel_size=1,
                                       pointwise_biases=pointwise_biases_2)

print(layer2_output.shape)
print(layer2_output)


