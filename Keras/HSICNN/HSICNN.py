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


################################
#按照数据预处理的格式，装载数据#
################################
def loadData(dataFile, typeId = -1, bShowData = False):
    data = sio.loadmat(dataFile)

    train_data = data['DataTr']
    train_label = data['CIdTr'][0,:]
#    train_label = train_label[0,:]
#    return train_data,train_label
    train_set = [train_data, train_label]

    test_data = data['DataTe']
    test_label = data['CIdTe'][0,:]
    test_set = [test_data, test_label]

    valid_data = data['DataVa']
    valid_label = data['CIdVa'][0,:]
    valid_set = [valid_data, valid_label]

#    def shared_dataset(data_xy, borrow=True):
#        data_x, data_y = data_xy
#        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX))
#        shared_y = theano.shared(numpy.asarray(data_y,dtype='int32'))
#        return shared_x, shared_y

#    test_set_x, test_set_y = shared_dataset(test_set)
#    valid_set_x, valid_set_y = shared_dataset(valid_set)
#    train_set_x, train_set_y = shared_dataset(train_set)

#    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    rval = [(train_data, train_label),(valid_data,valid_label),(test_data,test_label)]
    return rval



