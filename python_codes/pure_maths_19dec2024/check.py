import numpy as np
from train import *
def MAXPOOLING(input_img: list, shape: tuple, poolSize: tuple):
    width, height, channels = shape
    pxsize, pysize = poolSize
    output = np.zeros((width // pxsize, height // pysize, channels))

    for y in range(0, height, pysize):
        for x in range(0, width, pxsize):
            for ch in range(channels):
                max_val = float('-inf')  # Initialize to negative infinity
                for py in range(pysize):
                    for px in range(pxsize):
                        if (x + px) < width and (y + py) < height:
                            max_val = max(max_val, input_img[y + py][x + px][ch])
                output[y // pysize][x // pxsize][ch] = max_val
    print("shape after the pooling layer:", np.array(output).shape)
    return output.tolist()

def DEPTHWISE_SEP(inputImage: list, inputShape: tuple, layer:list , kernelShape: tuple, pointwise_shape: tuple) -> list:
    width = inputShape[0]
    height = inputShape[1]
    channelIn = inputShape[2]
    depthwise_kernel = layer[0]
    pointwise_kernel = layer[1]
    bias = layer[2]
    kxSize = kernelShape[0]
    kySize = kernelShape[1]
    channelOut = pointwise_shape[3]  # Output channels after pointwise convolution
    print("Depthwise Kernel Shape:", np.array(depthwise_kernel).shape)
    print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)

    # Step 1: Depthwise Convolution (each input channel has its own kernel)
    depthwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelIn))  # output of depthwise conv

    for y in range(height - kySize + 1):
        for x in range(width - kxSize + 1):
            for chIn in range(channelIn):
                for ky in range(kySize):
                    for kx in range(kxSize):
                        if ((y + ky) >= height) or ((x + kx) >= width):
                            continue
                        # Kernel value at [kx, ky, chIn, chOut]
                        kr = depthwise_kernel[kx][ky][chIn][0]
                        # Input pixel at [x + kx, y + ky, chIn]
                        px = inputImage[x + kx][y + ky][chIn]
                        # Add to the depthwise output for chIn
                        # print(f"kr: {kr}, type: {type(kr)}")
                        # print(f"px: {px}, type: {type(px)}")

                        depthwise_output[x][y][chIn] += kr * px

    # Step 2: Pointwise Convolution (1x1 convolution on depthwise output)
    pointwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelOut))  # output of pointwise conv
    # print("output after depthwise conv:")
    # # print(depthwise_output)
    # print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)

    for y in range(depthwise_output.shape[0]):
        for x in range(depthwise_output.shape[1]):
            for chOut in range(channelOut):
                for chIn in range(channelIn):
                    # Apply 1x1 pointwise convolution (mix all input channels)

                    kr = pointwise_kernel[0][0][chIn][chOut]
                    px = depthwise_output[x][y][chIn]

                    pointwise_output[x][y][chOut] += kr * px
    # pointwise_output += bias
    #print("output before adding the bias " , pointwise_output)
    pointwise_output += bias
    #print("output after adding the bias " , pointwise_output)
    return pointwise_output.tolist()

layer0_weights , layer2_weights , layer5_weights , layer6_weights = loadLayers()

testmodel = testData()
print("this is the input image")
print(testmodel)
layer0Out = DEPTHWISE_SEP(testmodel[0], (28, 28, 1), layer0_weights ,  (5,5,1,1) , (1,1,1,2))
layer0ReLU = RELU(layer0Out, (24, 24, 2))
layer1Out = MAXPOOLING(layer0ReLU, (24, 24, 2), (2, 2))
# print("shape after the pooling layer:" , np.array(layer1Out).shape)