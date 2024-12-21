import numpy as np
import json
from utility import RELU
#from utility import MAXPOOLING
from tensorflow import keras
#from utility import DENSE
from utility import flatten
from utility import DEPTHWISE_SEP
from utility import SOFTMAX
# def DEPTHWISE_SEP(inputImage: list, inputShape: tuple, layer:list , kernelShape: tuple, pointwise_shape: tuple) -> list:
#     width = inputShape[0]
#     height = inputShape[1]
#     channelIn = inputShape[2]
#     depthwise_kernel = layer[0]
#     pointwise_kernel = layer[1]
#     bias = layer[2]
#     kxSize = kernelShape[0]
#     kySize = kernelShape[1]
#     channelOut = pointwise_shape[3]
#     # print(width)
#     # print(height)
#     # print(channelIn)
#     # print(channelOut)
#     # print(np.array(depthwise_kernel).shape)
#     # print(pointwise_kernel)
#     # print(np.array(pointwise_kernel).shape)
#     # print(bias)
#     # print(kxSize)
#     # print(kySize)
#     # Output channels after pointwise convolution
#     # print("Depthwise Kernel Shape:", np.array(depthwise_kernel).shape)
#     # print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)
#
#     # Step 1: Depthwise Convolution (each input channel has its own kernel)
#     depthwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelIn))  # output of depthwise conv
#
#     for y in range(height - kySize + 1):
#         for x in range(width - kxSize + 1):
#             for chIn in range(channelIn):
#                 for ky in range(kySize):
#                     for kx in range(kxSize):
#                         if ((y + ky) >= height) or ((x + kx) >= width):
#                             continue
#                         # Kernel value at [kx, ky, chIn, chOut]
#                         #print("the indices are :" , kx , ky , chIn)
#
#                         kr = depthwise_kernel[kx][ky][chIn][0]
#                         # Input pixel at [x + kx, y + ky, chIn]
#                         # print("kernel " , kr)
#                         px = inputImage[x + kx][y + ky][chIn]
#                         # print("input image region " , px)
#                         # Add to the depthwise output for chIn
#                         # print(f"kr: {kr}, type: {type(kr)}")
#                         # print(f"px: {px}, type: {type(px)}")
#
#                         depthwise_output[x][y][chIn] += kr * px
#                         #print("output:" , depthwise_output[x][y][chIn])
#     # Step 2: Pointwise Convolution (1x1 convolution on depthwise output)
#     pointwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelOut))  # output of pointwise conv
#     # print("output after depthwise conv:")
#     # # print(depthwise_output)
#     # print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)
#
#     for y in range(depthwise_output.shape[0]):
#         for x in range(depthwise_output.shape[1]):
#             for chOut in range(channelOut):
#                 for chIn in range(channelIn):
#                     # Apply 1x1 pointwise convolution (mix all input channels)
#
#                     kr = pointwise_kernel[0][0][chIn][chOut]
#                     px = depthwise_output[x][y][chIn]
#                     print(kr, px)
#                     pointwise_output[x][y][chOut] += kr * px
#     # pointwise_output += bias
#     #print("output before adding the bias " , pointwise_output)
#     pointwise_output += bias
#     #print("output after adding the bias " , pointwise_output)
#     return pointwise_output.tolist()
def DENSE(inputimg: list, weights: list, shape: tuple) -> list:
    inlen, outlen = shape
    mulWeights = weights[0]
    biasWeights = weights[1]

    print("weights:", mulWeights, " ")
    print("Shape of weights:", np.array(mulWeights).shape)
    print("bias:", biasWeights)
    print("Shape of bias:", np.array(biasWeights).shape)
    print(inlen , outlen)
    output = np.zeros(outlen)

    # Iterate over each output neuron
    for i in range(outlen):
        sum_value = 0  # To store the summation for the output neuron
        print(f"Calculating output neuron {i}:")

        # Iterate over each input neuron
        for y in range(inlen):
            # Multiply the input neuron with the corresponding weight
            mul_value = inputimg[y] * mulWeights[y][i]
            sum_value += mul_value  # Add to the sum for the output neuron

            # Show the intermediate multiplication
            print(f"  input[{y}] * weight[{y}][{i}] = {inputimg[y]} * {mulWeights[y][i]} = {mul_value}")

        # Add the bias for the output neuron
        sum_value += biasWeights[i]
        print(f"  Adding bias: {biasWeights[i]}")

        # Set the final output value for the current neuron
        output[i] = sum_value
        print(f"  Final output value for neuron {i}: {output[i]}")

    print("Output after dense layer:", output)
    return output.tolist()
def flatten_column_major(inputImage: list) -> list:
    channels = len(inputImage)  # Number of channels
    height = len(inputImage[0])  # Height of the image
    width = len(inputImage[0][0])  # Width of the image
    #print(height , width , channels)
    flattened = []

    # Iterate through each spatial position (y, x)
    for ch in range(channels):
        for x in range(width):
            for y in range(height):


                flattened.append(inputImage[x][y][ch])

    return flattened
def MAXPOOLING(input_img: list, shape: tuple, poolSize: tuple):
    width, height, channels = shape
    pxsize, pysize = poolSize
    output = np.zeros((width // pxsize, height // pysize, channels))

    for y in range(0, height, pysize):
        for x in range(0, width, pxsize):
            for ch in range(channels):
                max_val = float('-inf')  # Initialize to negative infinity
                region = []  # To store the pooling region
                for py in range(pysize):
                    for px in range(pxsize):
                        if (x + px) < width and (y + py) < height:
                            region.append(input_img[y + py][x + px][ch])
                            max_val = max(max_val, input_img[y + py][x + px][ch])

                # Set the output value for the current pooling window
                output[y // pysize][x // pxsize][ch] = max_val

                # Print details for each pooling operation
                # print(
                #     f"Pooling Region (x: {x}-{x + pxsize}, y: {y}-{y + pysize}, ch: {ch}):\n{np.array(region).reshape(pysize, pxsize)}")
                # print(f"Max Value from Region: {max_val}\n")

    print("Shape after the pooling layer:", np.array(output).shape)
    return output.tolist()


# def DEPTHWISE_SEP(inputImage: list, inputShape: tuple, layer: list, kernelShape: tuple, pointwise_shape: tuple) -> list:
#     width, height, channelIn = inputShape
#     depthwise_kernel, pointwise_kernel, bias = layer
#     kxSize = kernelShape[0]
#     kySize = kernelShape[1]
#     channelOut = pointwise_shape[3]
#     inputImage = np.array(inputImage)
#     depthwise_kernel = np.array(depthwise_kernel)  # Convert to NumPy array
#     pointwise_kernel = np.array(pointwise_kernel)
#
#     # Step 1: Depthwise Convolution
#     depthwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelIn))
#     print("\nDepthwise Convolution:")
#
#     for y in range(height - kySize + 1):
#         for x in range(width - kxSize + 1):
#             for chIn in range(channelIn):
#                 region = inputImage[x:x + kxSize, y:y + kySize, chIn]
#                 kernel = depthwise_kernel[:, :, chIn, 0]
#                 element_wise_product = region * kernel
#                 output_value = np.sum(element_wise_product)
#                 depthwise_output[x, y, chIn] = output_value
#
#                 # Printing details
#                 print(f"Region (x: {x}-{x + kxSize}, y: {y}-{y + kySize}, chIn: {chIn}):\n{region}")
#                 print(f"Kernel (chIn: {chIn}):\n{kernel}")
#                 print(f"Element-wise product:\n{element_wise_product}")
#                 print(f"Summed output: {output_value}\n")
#
#     # Step 2: Pointwise Convolution
#     pointwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelOut))
#     print("\nPointwise Convolution:")
#
#     for y in range(depthwise_output.shape[1]):
#         for x in range(depthwise_output.shape[0]):
#             for chOut in range(channelOut):
#                 element_wise_product = depthwise_output[x, y, :] * pointwise_kernel[0, 0, :, chOut]
#                 output_value = np.sum(element_wise_product)
#                 pointwise_output[x, y, chOut] = output_value
#
#                 # Printing details
#                 print(f"Depthwise output at (x: {x}, y: {y}): {depthwise_output[x, y, :]}")
#                 print(f"Pointwise kernel (chOut: {chOut}): {pointwise_kernel[0, 0, :, chOut]}")
#                 print(f"Element-wise product:\n{element_wise_product}")
#                 print(f"Summed output: {output_value}\n")
#
#     # Adding bias
#     pointwise_output += bias
#     print("\nFinal Output after adding bias:\n", pointwise_output)
#
#     return pointwise_output.tolist()


def loadLayers():
    layer0_conv2d_weights = loadWeightsFromJson('./weights/json/layer0_conv2d.json')
    layer2_conv2d_weights = loadWeightsFromJson('./weights/json/layer2_conv2d.json')
    layer6_dense_weights = loadWeightsFromJson('./weights/json/layer6_dense.json')
    layer5_dense_weights = loadWeightsFromJson('./weights/json/layer5_dense.json')
    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights , layer6_dense_weights

def loadWeightsFromJson(path: str) -> list:
    with open(path, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data


def loadModel():
    model = keras.models.load_model('../trained_model.h5', compile=True)
    return model

def testData(i):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255 - 0.5
    x_test = x_test.astype("float32") / 255 - 0.5
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    testModel = x_test[i].reshape([1, 28, 28, -1])
    return testModel ,y_test[i]


def expectedAnswer(model, testModel):
    expected = model.predict(testModel)
    print(f"expected: {expected}")


def actualAnswer(layer0_conv2d_weights, layer2_conv2d_weights,layer5_dense_weights , layer6_dense_weights, testModel):

    layer0Out = DEPTHWISE_SEP(testModel[0], (28, 28, 1), layer0_conv2d_weights ,  (5,5,1,1) , (1,1,1,2))
    # print(layer0Out)
    # print(np.array(layer0Out).shape)
    layer0ReLU = RELU(layer0Out, (24, 24, 2))
    #print(layer0ReLU)
    #
    layer1Out = MAXPOOLING(layer0ReLU, (24, 24, 2), (2, 2))
    #print(layer1Out)
    #
    # print("second conv layer:::::")
    # print("#########################")
    layer2Out = DEPTHWISE_SEP(layer1Out, (12, 12, 2), layer2_conv2d_weights , (5,5,2,2) , (1,1,2,4))
    #print(layer2Out)
    #
    layer2ReLU = RELU(layer2Out, (8, 8, 4))
    #
    # print("####################################################################################")
    # print("####################################################################################")
    # print(layer2ReLU)
    layer3Out = MAXPOOLING(layer2ReLU, (8, 8, 4), (2, 2))
    #
    #print(layer3Out)
    layer4Flatten = flatten(layer3Out)
    print("flattened input neurons: ",layer4Flatten)
    print("flattened input" , layer4Flatten)
    layer5dense = DENSE(layer4Flatten , layer5_dense_weights , (64,64))
    print("before passing through pooling layer -> " , layer5dense)
    layer5pool = np.maximum(layer5dense,0)
    # # print("shape after flattening and passing thru dense layer ",np.array(layer5dense).shape)
    print("after the pooling layer")
    print(layer5pool)
    layer6dense = DENSE(layer5pool , layer6_dense_weights , (64,10))
    print("output after the last dense layer->>>>")
    print(layer6dense)
    output = SOFTMAX(layer6dense)
    max_index = output.index(max(output))
    print("the model predicted output is: " , max_index)

    return max_index

    #layer5Dropout = dropout(layer4Flatten, 0.5)



def main():

    model = loadModel()
    layer0_weights , layer2_weights , layer5_weights , layer6_weights = loadLayers()
    #testmodel, ans = testData()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255 - 0.5
    x_test = x_test.astype("float32") / 255 - 0.5
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    n = 1
    cnt = 0
    for i in range(n):
        testModel = x_test[i].reshape([1, 28, 28, -1])
        mode_ans = actualAnswer(layer0_weights, layer2_weights, layer5_weights ,layer6_weights, testModel)
        print("the real answer is :" , y_test[i])
        print("the model's ans is :" , mode_ans)
        if(y_test[i] == mode_ans):
            cnt+=1


    print("accuracy:" ,cnt)

if __name__ == "__main__":
    main()