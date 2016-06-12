#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD

from keras.utils import np_utils

##################################################
#This function is used to construct the CNN model#
##################################################
def build_CNN_model(layers, loss, optimizer):
    
    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.compile(loss=loss,optimizer=optimizer)
    
    return model

##########################################################
#This function is used to train the constructed CNN model#
##########################################################
def train_model(model, x_train, y _train, x_val, y_val, batch_size,nb_epoch):
    model.fit(x_train, y_train, batch_size = batch_size,
             nb_epoch = nb_epoch, show_accuracy=True,verbose=1,
             validation_data=(x_val, y_val))




