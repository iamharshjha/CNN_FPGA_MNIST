import numpy as np
from utilities import *
from layer2_conv import *
from tensorflow.python.ops.numpy_ops.np_array_ops import flatten

from python_codes.pure_maths.layer2_conv import process_layer2

""" 
so here first I am reading the weights which I extracted from extract_weights.py 
since the input was not in the correct format we first flatten it and then process it for further usage

"""
input_img = np.random.rand(28,28,1)
output_conv2_layer = process_layer2(input_img)
output_layer_5 = process_layer5_file('../weights/layer5_dense_weights.txt', 'layer5_flat.txt',output_conv2_layer)
print(output_layer_5.shape)




