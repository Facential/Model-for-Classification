# Machine Learning Image Classification Python Notebook

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

After running the notebook, you will obtain the following results:



