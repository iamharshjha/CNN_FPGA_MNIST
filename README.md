# CNN_FPGA_MNIST
Part of Mtech project


The aim of the project is to implement depthwise separable convolution on FPGA board ( not yet decided the platform to use). For that we need to implement a verilog code which reads the weights and biases and do the 
required operations on the input images. 

So to start , we first developed a python model using the Tensorflow library and trained it on MNIST dataset. 
After training the model , I saved the model trained_model.h5

Then in the folder pure_maths_19dec2024 , we extracted weights as json ( to read it for our raw code) and as txt file so that we can understand the shape of the weights. 

Then in utility.py we have defined functions like -> softmax , dense , maxpooling , depthwise_sepconv etc....

check,py was to test the model after each layer. 
The integration of all layers was done in train_raw.py

All the references of this project is from this git repo : https://github.com/XAli-SHX/FPGA-Implementation-of-Image-Processing-for-MNIST-Dataset-Based-on-CNN-Algorithm

Only difference is in this repo , standard convolution was used and in this one we have used depthwise separable convolution , which reduces the number of multiplications
 
