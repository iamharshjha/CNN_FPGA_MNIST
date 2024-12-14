import numpy as np
import json
from utilities import *
def load_weights_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def depthwise_sep_conv(image: np.ndarray, depthwise_weights: np.ndarray, pointwise_weights: np.ndarray,
                       depthwise_kernel_size: int, pointwise_kernel_size: int, pointwise_biases: np.ndarray) -> np.ndarray:
    """
    Perform depthwise separable convolution on an image using the provided weights and biases.

    :param image: Input image of shape (height, width, channels)
    :param depthwise_weights: Depthwise convolution weights of shape (channels, kernel_size, kernel_size)
    :param pointwise_weights: Pointwise convolution weights of shape (1, 1, channels, num_output_channels)
    :param depthwise_kernel_size: Size of the depthwise convolution kernel (assumed square)
    :param pointwise_kernel_size: Size of the pointwise convolution kernel (always 1x1)
    :param pointwise_biases: Biases for the pointwise convolution
    :return: Output after depthwise separable convolution
    """
    input_height, input_width, input_channels = image.shape

    # Depthwise convolution
    depthwise_output = np.zeros((input_height - depthwise_kernel_size + 1, input_width - depthwise_kernel_size + 1, input_channels))

    for y in range(input_height - depthwise_kernel_size + 1):
        for x in range(input_width - depthwise_kernel_size + 1):
            for c in range(input_channels):
                region = image[y:y + depthwise_kernel_size, x:x + depthwise_kernel_size, c]
                kernel = depthwise_weights[c]  # depthwise kernel for this channel
                depthwise_output[y, x, c] = np.sum(region * kernel)  # Depthwise convolution output

    output_height, output_width, _ = depthwise_output.shape

    # Pointwise convolution (1x1 convolution)
    pointwise_output = np.zeros((output_height, output_width, pointwise_weights.shape[-1]))

    for y in range(output_height):
        for x in range(output_width):
            for c_out in range(pointwise_output.shape[2]):
                # Pointwise convolution: summing the depthwise output weighted by the pointwise kernel
                pointwise_output[y, x, c_out] = np.sum(depthwise_output[y, x, :] * pointwise_weights[0, 0, :, c_out]) + pointwise_biases[c_out]

    return max_pooling(pointwise_output)



"""if __name__ == "__main__":
    # Load weights and biases from a JSON file
    weights_file_path = '../weights/layer0_separable_conv2d.json'  # Replace with your JSON file path
    weights_data = load_weights_from_json(weights_file_path)

    # Extract weights and biases from the flat list
    depthwise_weights = np.array(weights_data[:25]).reshape((5, 5, 1))  # 5x5 kernel for 1 channel
    pointwise_weights = np.array(weights_data[25:27]).reshape((1, 1, 1, 2))  # 1x1 kernel for 2 output channels
    pointwise_biases = np.array(weights_data[27:29])  # Biases for the pointwise convolution

    #print("Depthwise weights:")
    #print(depthwise_weights)
    #print("Pointwise weights:")
    #print(pointwise_weights)
    #print("Pointwise biases:")
    #print(pointwise_biases)

    # Sample input image (e.g., a 28x28 image with 1 channel)
    #input_image = np.random.rand(28, 28, 1)  # Example input, replace with your actual image

    # Perform depthwise separable convolution
    output = depthwise_sep_conv(input_image, depthwise_weights, pointwise_weights,
                                depthwise_kernel_size=5, pointwise_kernel_size=1,
                                pointwise_biases=pointwise_biases)
    #output = max_pooling(output)
    #print("shape:", output.shape)
    #print(type(output))"""

def process_layer1(input_image: np.ndarray) -> np.ndarray:
    """
    Process the input image through layer 1 and return the output.
    """
    weights_file_path = '../weights/layer0_separable_conv2d.json'
    weights_data = load_weights_from_json(weights_file_path)

    # Extract weights and biases for layer 1
    depthwise_weights = np.array(weights_data[:25]).reshape((5, 5, 1))
    pointwise_weights = np.array(weights_data[25:27]).reshape((1, 1, 1, 2))
    pointwise_biases = np.array(weights_data[27:29])

    # Perform depthwise separable convolution for layer 1
    layer1_output = depthwise_sep_conv(input_image, depthwise_weights, pointwise_weights,
                                       depthwise_kernel_size=5, pointwise_kernel_size=1,
                                       pointwise_biases=pointwise_biases)
    print("shape after layer 1 :" , layer1_output.shape)
    return layer1_output