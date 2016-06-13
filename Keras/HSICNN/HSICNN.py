#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD

from keras.utils import np_utils

import math

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

#######################################################################################
#currently, I wrote all the network constructing and training and testing in this file#
#laterly, I will seperate them apart.                                                 #
#######################################################################################
def temp_network(filePath, number_of_con_filters = 20, con_step_length, max_pooling_feature_map_size, number_of_full_layer_nodes, learning_ratio, train_decay):
    #get the train data, train label, validate data, validate label, test data, test label
    train_dataset, valid_dataset, test_dataset = loadData(filePath)
    
    #initialize parameters
    layer1_input_length = len(test_dataset[0])
    con_filter_length = (math.ceil( (layer1_input_length /  con_step_length) / 9)) * con_step_length

    model = Sequential()
    
    #the first convolutional layer
    print 'The size of the first convolutional layer is %d.' % layer1_input_length
    layer1 = Convolution1D(number_of_con_filters, con_filter_length, activation = tanh, border_mode='same', bias=True, input_dim=layer1_input_length)
    model.add(layer1)

    #the max pooling layer after the first convolutional layer
    first_feature_map_size = (layer1_input_length - con_filter_length) / con_step_length + 1
    max_pooling_kernel_size = math.ceil(first_feature_map_size / max_pooling_feature_map_size)
    layer2 = MaxPooling1D(max_pooling_kernel_size)
    model.add(layer2)

    #Flatten the variables outputed from maxpooling layer
    model.add(Flatten())

    #the fully connected layer
    layer3 = Dense(number_of_full_layer_nodes, activation = tanh, bias = True)
    model.add(layer3)

    #the activation layer which will output the final classification result
    classes = numpy.max(test_dataset[1])
    print 'The number of classification classes is %d.' % classes
    layer4 = Dense(classes,activation = softmax, bias=True)
    model.add(layer4)

    #the optimizer
    sgd = SGD(lr = learning_ratio, decay = train_decay, momentum = 0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    #train the constructed model
    history = model.fit(self, train_dataset[0], train_dataset[1], batch_size = 10, nb_epoch = 1000, verbose=1, valid_dataset, shuffle=True)
    model.save_weights('model_weights.h5', overwrite=True)

    #test the model
    classes = model.predict_classes(test_dataset[0], verbose=1)
    test_accuracy = numpy.mean(numpy.equal(test_dataset[1], classification))

    print 'The correct ratio of the trained CNN model is %d' % test_accuracy
