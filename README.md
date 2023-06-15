# Machine Learning Face Skin Type Classification Python Notebook

This repository contains some Python notebooks that demonstrate image classification using machine learning techniques. The notebook is written in Python and utilizes popular libraries such as TensorFlow and Keras. There is not just one notebook that we put here as a result of trying different pre-trained models that we will going to choose to implement. Each notebook has a different pre-trained model used to make our face skin type classifier, but after doing some research we found that VGGFace2 suits us the best for our application compared to InceptionV3, VGG16, and ResNET50. Because of that, we decided to develop just the notebook with the file name facentialClassification_VGGFace2.ipnyb as our main notebook for building our model.

This is what vggFace pre-trained model looks like in a full architecture without any layer modified
```
Model: "model_1"
_________________________________________________________________
Layer (type)            Output Shape                Param #
=================================================================
input_3 (InputLayer)    [(None, 224, 224, 3)]       0

conv1_1 (Conv2D)        (None, 224, 224, 64)        1792

conv1_2 (Conv2D)        (None, 224, 224, 64)        36928

pool1 (MaxPooling2D)    (None, 112, 112, 64)        0

conv2_1 (Conv2D)        (None, 112, 112, 128)       73856

conv2_2 (Conv2D)        (None, 112, 112, 128)       147584

pool2 (MaxPooling2D)    (None, 56, 56, 128)         0

conv3_1 (Conv2D)        (None, 56, 56, 256)         295168

conv3_2 (Conv2D)        (None, 56, 56, 256)         590080

conv3_3 (Conv2D)        (None, 56, 56, 256)         590080

pool3 (MaxPooling2D)    (None, 28, 28, 256)         0

conv4_1 (Conv2D)        (None, 28, 28, 512)         1180160

conv4_2 (Conv2D)        (None, 28, 28, 512)         2359808

conv4_3 (Conv2D)        (None, 28, 28, 512)         2359808

pool4 (MaxPooling2D)    (None, 14, 14, 512)         0

conv5_1 (Conv2D)        (None, 14, 14, 512)         2359808

conv5_2 (Conv2D)        (None, 14, 14, 512)         2359808

conv5_3 (Conv2D)        (None, 14, 14, 512)         2359808

flatten_1 (Flatten)     (None, 100352)              0

dense_2 (Dense)         (None, 1024)                102761472

dropout_1 (Dropout)     (None, 1024)                0

dense_3 (Dense)         (None, 6)                   6150
=================================================================
Total params: 117,482,310
Trainable params: 102,767,622
Non-trainable params: 14,714,688
_________________________________________________________________
```
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

The dataset used in this notebook came from different resources because we cannot find a single public dataset focussing on human face skin types. After some discussion, finally we picked one dataset from kaggle.com that is not yet specific for human face skin type, but already has a very clear and representative human face with different races and is also close to what our users are going to look like when taking their faces using a smartphone's front camera. And then to match our project use case, we assigned that dataset that contains images to our predefined face skin types, which are dry, oily, normal, combination, sensitive, and one extra class for detecting a non-face object. 

So, after we labeled the datasets it consists of 1290 images belonging to 6 different classes which are dry, oily, normal, combination, sensitive, and nonface. The dataset is divided by 80 percent into training and 20 percent into testing sets for model evaluation.

This is the link to kaggle open-source dataset that contributes the most to our redefined dataset.
https://www.kaggle.com/datasets/tapakah68/skin-problems-34-on-the-iga-scale?select=images




## Results

After doing experiments, these are the report of what we got so far. In the next section, we will explain through the whole code, how many model versions we have, how many notebook versions we created, the architectures we tried during the experiment, etc. And for quick context, we already commit the version we are going to use for our app in the latest version or the latest commit on this repository.


# Creating Model

We have done a few attempts using different architecture to be used for our machine learning model training process in order to get the best-performing model.

* This is our first architecture 

In this version, we used vggFace pre-trained model and at the last layer, we add a convolution layer with 1024 neuron units with ReLU as activation. After that, to fit the output desired for our classification we add a dense layer with 1024 neuron units with ReLU activation. then add a layer of dropout with a value of 0.2 to minimize overfitting. And lastly the output layer with 6 neuron units for our model prediction. 

```
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
=======================================================================
Total params: 70,821,702
Trainable params: 56,107,014
Non-trainable params: 14,714,688
_______________________________________________________________________
```

With that architecture we got a model that perform poorly, this pattern looks like our model has experienced an overfitting. After doing 150 epoch with 0.0001 learning rate using Adam optimizer we got accuracy that perform similar to other architectures that we have tried, but very bad at validation loss value. Further, the validation loss keeps increasing as a sign that this architecture is generating an overfitting model. So we did not use this model version to our app.

![image](https://github.com/Facential/Model-for-Classification/assets/70127988/277dc533-217d-48bd-920c-6a8d107120e1)  ![image](https://github.com/Facential/Model-for-Classification/assets/70127988/f841fd3c-a991-4544-ad54-af8703734a22)

In order to reduce overfitting we tried adjusting the hyperparameter again with the idea to reducing the complexity of the architecture so that our model can be more generalized. To do that we make a new architecture without that extra convolution layer which has 1024 neuron units. And then to further generalized our model we cut the vggFace pre-trained model, which means we are not using the full architecture of that vggFace pre-trained model. We made a cutoff from the last two layers of that vggFace pre-trained model, which is the 'conv5_3' layer, so the 'conv5_3' convolution layer will be our last layer from just the vggFace pre-trained. The rest is still the same like the previous architecture, we add a dense layer with 1024 neuron units with ReLU activation, then add a layer of dropout with a value of 0.2 to minimize overfitting. And lastly the output layer with 6 neuron units for our model prediction. Here is the complete architecture of our second model

```
Model: "model_1"
_________________________________________________________________
Layer (type)            Output Shape                  Param #
=================================================================
input_3 (InputLayer)    [(None, 224, 224, 3)]         0

conv1_1 (Conv2D)        (None, 224, 224, 64)          1792

conv1_2 (Conv2D)        (None, 224, 224, 64)          36928

pool1 (MaxPooling2D)    (None, 112, 112, 64)          0

conv2_1 (Conv2D)        (None, 112, 112, 128)         73856

conv2_2 (Conv2D)        (None, 112, 112, 128)         147584

pool2 (MaxPooling2D)    (None, 56, 56, 128)           0

conv3_1 (Conv2D)        (None, 56, 56, 256)           295168

conv3_2 (Conv2D)        (None, 56, 56, 256)           590080

conv3_3 (Conv2D)        (None, 56, 56, 256)           590080

pool3 (MaxPooling2D)    (None, 28, 28, 256)           0

conv4_1 (Conv2D)        (None, 28, 28, 512)           1180160

conv4_2 (Conv2D)        (None, 28, 28, 512)           2359808

conv4_3 (Conv2D)        (None, 28, 28, 512)           2359808

pool4 (MaxPooling2D)    (None, 14, 14, 512)           0

conv5_1 (Conv2D)        (None, 14, 14, 512)           2359808

conv5_2 (Conv2D)        (None, 14, 14, 512)           2359808

conv5_3 (Conv2D)        (None, 14, 14, 512)           2359808

flatten_1 (Flatten)     (None, 100352)                0

dense_2 (Dense)         (None, 1024)                  102761472

dropout_1 (Dropout)     (None, 1024)                  0

dense_3 (Dense)         (None, 6)                     6150
=================================================================
Total params: 117,482,310
Trainable params: 102,767,622
Non-trainable params: 14,714,688
_________________________________________________________________

```

Using our new architecture we got a model that perform better and the fact is after all the experiment this model result is the best we could get, any changes turn out to perform worse or not improve at all. Still, this pattern looks like our model is not perfect yet and experienced an underfitting. After doing 150 epochs with a 0.0001 learning rate using Adam optimizer we got accuracy that perform similar to other architectures that we have tried, and we managed to keep the validation loss converge to keep decrease, the negative point is validation loss is converging not as fast as the loss itself. This happened because our datasets that still lacking.

![image](https://github.com/Facential/Model-for-Classification/assets/70127988/d9bffffc-648c-4c95-b58e-0bfa7586e8a5)  ![image](https://github.com/Facential/Model-for-Classification/assets/70127988/4cbf4440-12d5-4402-9043-587251267d83)
 
As we can see both versions can easily reach around 0.9 in accuracy and fall behind by about 0.7 in validation accuracy. But in the second model, our validation loss perform better for not climbing up high when the epoch is higher compared to the first model.

Here are some of the history of models we created and decided not going to be used

For more information the model that we implemented to our apps is the model in a keras format with name **model_vggFace_11Juni.h5**
![image](https://github.com/Facential/Model-for-Classification/assets/70127988/7acdd2c0-27bb-4b2e-a08d-d7c2311b01bb)

![image](https://github.com/Facential/Model-for-Classification/assets/70127988/3a4d8cb2-12c9-4fac-afc9-d04121ac8214)

 





