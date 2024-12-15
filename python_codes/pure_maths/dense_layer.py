import numpy as np
from utilities import *
from layer2_conv import *
from tensorflow.python.ops.numpy_ops.np_array_ops import flatten

from python_codes.pure_maths.layer2_conv import process_layer2
from tensorflow.keras.datasets import mnist
""" 
so here first I am reading the weights which I extracted from extract_weights.py 
since the input was not in the correct format we first flatten it and then process it for further usage

"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()
n = 100
cnt = 0
for i in range(n):
# Select a random image from the dataset, for example, the first image
    input_img = x_test[i]  # Choose an image from the training set

    # Normalize the image to match the range of values in your example (0-1)
    input_img = input_img.astype('float32') / 255.0  # Scaling the values to [0, 1]

    # Reshape to add the channel dimension (28x28x1)
    input_img = np.expand_dims(input_img, axis=-1)  # Adding the channel dimension (1 for grayscale)

    print("Shape of input image:", input_img.shape)



    #input_img = np.random.rand(28,28,1)
    output_conv2_layer = process_layer2(input_img)
    output_layer_5 = process_layer5_file('../weights/layer5_dense_weights.txt', 'layer5_flat.txt',output_conv2_layer)
    print(output_layer_5.shape)
    x = process_layer_6_file('layer_6_flat.txt',output_layer_5)
    final_output = softmax(x)
    print("softmax output: " , final_output)
    op = np.argmax(final_output,axis = -1)
    print(op)
    print(y_test[i])
    if(op == y_test[i]):
        cnt+=1

print("accuracy" , cnt)



