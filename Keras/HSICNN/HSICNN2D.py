#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD
#import imdb

import theano

#from keras.processing import sequence
#from keras.layers.Activation import tanh, softmax

from keras.utils import np_utils

import scipy.io as sio
import random
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
#def train_model(model, x_train, y _train, x_val, y_val, batch_size,nb_epoch):
#    model.fit(x_train, y_train, batch_size = batch_size,
#             nb_epoch = nb_epoch, show_accuracy=True,verbose=1,
#             validation_data=(x_val, y_val))


################################
#按照数据预处理的格式，装载数据#
################################
def loadData(dataFile, typeId = -1, bShowData = False):
    data = sio.loadmat(dataFile)

    train_data = data['DataTr']
    train_label_temp = data['CIdTr'][0,:]
#    train_label = train_label[0,:]
#    return train_data,train_label
#    train_set = [train_data, train_label]

    test_data = data['DataTe']
    test_label_temp = data['CIdTe'][0,:]
#    test_set = [test_data, test_label]

    valid_data = data['DataVa']
    valid_label_temp = data['CIdVa'][0,:]
#    valid_set = [valid_data, valid_label]

#    def shared_dataset(data_xy, borrow=True):
#        data_x, data_y = data_xy
#        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX))
#        shared_y = theano.shared(numpy.asarray(data_y,dtype='int32'))
#        return shared_x, shared_y

#   test_set_x, test_set_y = shared_dataset(test_set)
#    valid_set_x, valid_set_y = shared_dataset(valid_set)
#    train_set_x, train_set_y = shared_dataset(train_set)

#   rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    

#    train_dataset_data = train_data.tolist()
#    test_dataset_data = test_data.tolist()
#    valid_dataset_data = valid_data.tolist()
    
    train_label = numpy.empty(len(train_label_temp))
    valid_label = numpy.empty(len(valid_label_temp))
    test_label = numpy.empty(len(test_label_temp))

    train_dataset_data = []
    nX = []
    count = 0
    for x in train_data:
        nx = []
        for w in x:
            nx.append(w)
        numpy.array(nx,dtype="object",copy=True)
        nX.append(nx)
        train_label[count] = train_label_temp[count]
        count = count + 1
    train_dataset_data = nX
#    testTemp = numpy.array(nX[:len(nX)])

    valid__dataset_data = []
    nX = []
    count = 0
    for x in valid_data:
        nx = []
        for w in x:
            nx.append(w)
        numpy.array(nx,dtype="object",copy=True)
        nX.append(nx)
        valid_label[count] = valid_label_temp[count]
        count = count + 1
    valid_dataset_data = nX


    test_dataset_data = []
    nX = []
    count = 0
    for x in test_data:
        nx = []
        for w in x:
            nx.append(w)
        numpy.array(nx,dtype="object",copy=True)
        nX.append(nx)
        test_label[count] = test_label_temp[count]
        count = count + 1
    test_dataset_data = nX

    train_dataset_data = numpy.array(train_dataset_data,dtype="object")
    test_dataset_data = numpy.array(test_dataset_data)
    valid_dataset_data = numpy.array(valid_dataset_data)

    return [(train_dataset_data, train_label),(valid_dataset_data,valid_label),(test_dataset_data,test_label)]
#    return rval

#######################################################################################
#currently, I wrote all the network constructing and training and testing in this file#
#laterly, I will seperate them apart.                                                 #
#######################################################################################
def temp_network(filePath, number_of_con_filters, con_step_length, max_pooling_feature_map_size, number_of_full_layer_nodes, learning_ratio, train_decay):
    #get the train data, train label, validate data, validate label, test data, test label
    train_dataset, valid_dataset, test_dataset = loadData(filePath)


    #the dimension of the input signal's chanel
    channel_length = train_dataset[0].shape[1]
    sample_counts = train_dataset[0].shape[0]

#    train_dataset, test_dataset = imdb.load_data()   
    #initialize parameters
    layer1_input_length = len(test_dataset[0][0])
    con_filter_length = int((math.ceil( (layer1_input_length /  con_step_length) / 9)) * con_step_length)

    destinations = numpy.max(test_dataset[1])
    
    #############################
    #Network Information Display#
    #############################
    print("The network have ", channel_length, "input nodes in the 1st layer.")
    print("The amount of samples in the dataset is ", sample_counts)
    print("The number of classification classes is ", destinations)
    print("The size of the first convolutional layer is ", layer1_input_length)
    print('The number of convolutional filters is ', number_of_con_filters, ",each kernel sizes ", con_filter_length,"X1.")
    print("There are ",number_of_full_layer_nodes," nodes in the fully connect layer.")
    #########################
    #Construct the CNN model# 
    #########################
    
    model = Sequential()
    
    #the first convolutional layer
    layer1 = Convolution2D(number_of_con_filters,nb_row = con_filter_length, nb_col = 1,border_mode='valid', subsample=(1,1),dim_ordering='th', bias=True,input_shape=(1,layer1_input_length, 1))

    print("The input to the first convolutional layer shapes", (1,layer1_input_length,1))
    model.add(layer1)

    model.add(Activation('tanh'))



    #the max pooling layer after the first convolutional layer
    first_feature_map_size = (layer1_input_length - con_filter_length) / con_step_length + 1
    max_pooling_kernel_size = int(math.ceil(first_feature_map_size / max_pooling_feature_map_size))
    print("The max pooling kernel size is ", max_pooling_kernel_size)
    layer2 = MaxPooling2D(pool_size = (max_pooling_kernel_size,1), strides=(max_pooling_kernel_size,1), border_mode='valid',dim_ordering='th')
    model.add(layer2)

    #Flatten the variables outputed from maxpooling layer
    model.add(Flatten())
    
    #the fully connected layer
    layer3 = Dense(number_of_full_layer_nodes, bias = True)
    model.add(layer3)
    model.add(Activation('tanh'))

    #the activation layer which will output the final classification result
    layer4 = Dense(destinations + 1,activation = 'tanh', bias=True)
#    layer4 = Activation('tanh')
    model.add(layer4)

    layer5 = Activation('softmax')
    model.add(layer5)

    #the optimizer
    sgd = SGD(lr = learning_ratio, decay = train_decay, momentum = 0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    #train the constructed model
    #the input shape of the train_dataset should be a list.


#    train_dataset_data = train_dataset[0].tolist()
#    test_dataset_data = test_dataset[0].tolist()

#    train_dataset_data = []
#    for tempTrainMark in range(train_dataset[0].shape[0]):
#        train_dataset_data.append(train_dataset[0][tempTrainMark])

#   train_dataset_data = numpy.array(train_dataset_data)
    
#    print("The training dataset have ", len(train_dataset_data), "items, each item is a ", len(train_dataset_data[0]), "dimention vector.")
#    print("There are ", len(train_dataset[1]),"labels in training dataset, the match between labels and item numbers is:", len(train_dataset[1]) == len(train_dataset_data), ".")
#    test_dataset_data = []
#    for tempTestMark in range(test_dataset[0].shape[0]):
#        test_dataset_data.append(test_dataset[0][tempTestMark])


#    test_dataset_data = numpy.array(test_dataset_data)
#    train_dataset_data = numpy.array(train_dataset_data[:len(train_dataset_data)])
#    test_dataset_data = numpy.array(test_dataset_data[:len(test_dataset_data)])

#    train_dataset_data = sequence.pad_sequences(train_dataset_data, )

#    train_dataset_data = train_dataset[0]
    

#    train_dataset_data = numpy.expand_dims(train_dataset[0], 1)
    train_dataset_data = train_dataset[0].reshape(train_dataset[0].shape[0],1,train_dataset[0].shape[1],1)
    train_dataset_label = np_utils.to_categorical(train_dataset[1])
    print("The dataset used to train the model shapes ",train_dataset_data.shape)
    print("The label corresponding to the train data shapes ",train_dataset_label.shape)
#    print(train_dataset_data)
    history = model.fit(train_dataset_data, train_dataset_label, batch_size = 10, nb_epoch = 1000, verbose=1, validation_split = 0.1, shuffle=True)
    model.save_weights('model_weights.h5', overwrite=True)

    #test the model
    classes = model.predict_classes(test_dataset[0], verbose=1)
    test_accuracy = numpy.mean(numpy.equal(test_dataset[1], classification))

    print("The correct ratio of the trained CNN model is ",  test_accuracy)

if __name__ == '__main__':
    temp_network("newPU1NWith2RestN.mat", number_of_con_filters = 20, con_step_length = 1, max_pooling_feature_map_size = 40, number_of_full_layer_nodes = 100, learning_ratio = 0.01, train_decay = 0.001)
