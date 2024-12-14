import numpy as np

def max_pooling(input_array, pool_size=(2, 2), stride=(2, 2)):

    H, W, C = input_array.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride

    # Calculate output dimensions
    out_height = (H - pool_height) // stride_height + 1
    out_width = (W - pool_width) // stride_width + 1

    # Initialize the output array
    output = np.zeros((out_height, out_width, C))

    for c in range(C):
        for i in range(0, out_height * stride_height, stride_height):
            for j in range(0, out_width * stride_width, stride_width):
                # Apply max pooling for each channel
                output[i // stride_height, j // stride_width, c] = np.max(
                    input_array[i:i+pool_height, j:j+pool_width, c]
                )

    return output

def softmax(x):
    # Subtracting the max value for numerical stability
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)

