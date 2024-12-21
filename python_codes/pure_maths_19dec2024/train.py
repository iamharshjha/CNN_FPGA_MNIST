import numpy as np
import json
from utility import *
from tensorflow import keras

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

def testData():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255 - 0.5
    x_test = x_test.astype("float32") / 255 - 0.5
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    testModel = x_test[6].reshape([1, 28, 28, -1])
    return testModel ,y_test[6]


def expectedAnswer(model, testModel):
    expected = model.predict(testModel)
    print(f"expected: {expected}")


def actualAnswer(layer0_conv2d_weights, layer2_conv2d_weights,layer5_dense_weights , layer6_dense_weights, testModel):

    layer0Out = DEPTHWISE_SEP(testModel[0], (28, 28, 1), layer0_conv2d_weights ,  (5,5,1,1) , (1,1,1,2))

    layer0ReLU = RELU(layer0Out, (24, 24, 2))

    layer1Out = MAXPOOLING(layer0ReLU, (24, 24, 2), (2, 2))

    layer2Out = DEPTHWISE_SEP(layer1Out, (12, 12, 2), layer2_conv2d_weights , (5,5,2,2) , (1,1,2,4))

    layer2ReLU = RELU(layer2Out, (8, 8, 4))

    layer3Out = MAXPOOLING(layer2ReLU, (8, 8, 4), (2, 2))

    layer4Flatten = flatten(layer3Out)
    layer5dense = DENSE(layer4Flatten , layer5_dense_weights , (64,64))
    layer5pool = np.maximum(layer5dense,0)
    # print("shape after flattening and passing thru dense layer ",np.array(layer5dense).shape)
    layer6dense = DENSE(layer5pool , layer6_dense_weights , (64,10))
    output = SOFTMAX(layer6dense)
    max_index = output.index(max(output))
    print("the model predicted output is: " , max_index)

    return max_index

    #layer5Dropout = dropout(layer4Flatten, 0.5)



def main():

    model = loadModel()
    n = 200
    cnt = 0
    layer0_weights , layer2_weights , layer5_weights , layer6_weights = loadLayers()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255 - 0.5
    x_test = x_test.astype("float32") / 255 - 0.5
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    for i in range(n):

        testModel = x_test[i].reshape([1, 28, 28, -1])
        # testmodel, ans = testData()
        ans = actualAnswer(layer0_weights, layer2_weights, layer5_weights ,layer6_weights, testModel)
        print("actual answer:" ,y_test[i])
        if (y_test[i] == ans):
            cnt+=1

    print("accuracy ->" , (cnt/n)*100)



if __name__ == "__main__":
    main()