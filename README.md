# Machine Learning Face Skin Type Classification Python Notebook

This repository contains some Python notebooks that demonstrates image classification using machine learning techniques. The notebook is written in Python and utilizes popular libraries such as TensorFlow and Keras. There are not just one notebook that we put here as a result of trying different pretrained model that we will going to choose to implement. On each notebook has a different pretrained model used to make our face skin type classifier, but after doing some research we found that VGGFace2 is suits the best for our applicaiton compared to InceptionV3, VGG16, and ResNET50. Because of that we decided to develope just the notebook with file name facentialClassification_VGGFace2.ipnyb as our main notebook for building our model.

## Overview

The purpose of this notebook is trying to explain the process of building and training an image classification model for our application use case. The model is trained on a dataset of images and is capable of predicting the class labels of unseen images.

The notebook covers the following key steps:

1. Dataset preparation and preprocessing
2. Model architecture design
3. Model training and evaluation
4. Model testing and predictions

## Requirements

In this notebook, these are some packages and libraries that needed and installed to our notebook:

- Python (version 3.10.10)
- TensorFlow (version 2.11.1)
- Keras (version 2.11.0)
- NumPy
- Matplotlib
- etc.


## Dataset

The dataset used in this notebook is came from different resources, because we cannot find a single public dataset focussing on human face skin types. After some discussion, finally we picked one dataset from kaggle.com that is not yet specific for human face skin type, but already has a very clear and representative human face with different races and also close to what our users are going to look like when taking their faces using a smartphone's front camera. And then to match our project use case, we assigned that dataset that contains images to our predefined face skin types, which are dry, oily, normal, combination, sensitive, and one extra class for detecting a non-face object. 

So, after we labeled the datasets it consists of 1290 images belonging to 6 different classes which are dry, oily, normal, combination, sensitive, and nonface. The dataset is divided by 80 percent into training and 20 percent into testing sets for model evaluation.

This is the link to kaggle open source dataset that contribute the most to our redefined dataset.
https://www.kaggle.com/datasets/tapakah68/skin-problems-34-on-the-iga-scale?select=images




## Results

After doing experiments, these are the report of what we got so far. In the next section we will explain trough the whole code, how many model version we have, how many notebook version we created, architectures we tried during experiment, etc. And for quick context, we already commit the version we are going to use for our app in the latest version or latest commit on this repository.


# Creating Model

We have done few attempts using different acrhitecture to be used for our machine learning model training process in order to get the best performing model.

* This is our first architecture 

In this version we used vggFace pretrained model and at the last layer we add a convolution layer with 1024 neuron units with ReLU as activation. After that, to fit the output desired for our classification we add a densed layer with 1024 neuron unit with ReLU activation. then add a layer of dropout with a value 0.2 to minimize overfitting. And lastly the output layer with 6 neuron unit for our model prediction. 

Model: "model"
_______________________________________________________________________
Layer (type)                      Output Shape                Param #
=======================================================================
vggface_vgg16_input (InputLayer) [(None, 224, 224, 3)]        0
vggface_vgg16 (Functional)        (None, 7, 7, 512)           14714688
conv2d (Conv2D)                   (None, 7, 7, 1024)          4719616
flatten (Flatten)                 (None, 50176)               0
dense (Dense)                     (None, 1024)                51381248
dropout (Dropout)                 (None, 1024)                0
dense_1 (Dense)                   (None, 6)                   6150
=================================================================
Total params: 70,821,702
Trainable params: 56,107,014
Non-trainable params: 14,714,688
_________________________________________________________________


with that architecture we got a model that perform poorly

After doing 150 epoch with 0.0001 learning rate using Adam optimizer we got Accuracy that perform simmilar to other architecture that we have tried. But very bad performance at validation loss value. Further the loss is keep increasing as a sign that this architecture is generate an overfitting model. So we did not use this model version to our app.

![image](https://github.com/Facential/Model-for-Classification/assets/70127988/277dc533-217d-48bd-920c-6a8d107120e1)  ![image](https://github.com/Facential/Model-for-Classification/assets/70127988/f841fd3c-a991-4544-ad54-af8703734a22)






