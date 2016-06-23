#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import numpy

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Convolution3D, MaxPooling3D
from keras.optimizers import SGD
#import imdb
#from keras.processing import sequence
#from keras.layers.Activation import tanh, softmax

from keras.utils import np_utils

import scipy.io as sio
import random
import math
from keras import backend as K

from sklearn import cross_validation,decomposition,svm

import time
def getMiddleOutPut(model,inputVector,kthlayer):
    getFunc = K.function([model.layers[0].input],[model.layers[kthlayer].output])
    layer_output = getFunc(inputVector)[0]
    return layer_output

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

    valid_data = data['DataTr']
    valid_label_temp = data['CIdTr'][0,:]
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
        train_label[count] = int(train_label_temp[count])
        count = count + 1
    train_dataset_data = nX
#    testTemp = numpy.array(nX[:len(nX)])

    valid_dataset_data = []
    nX = []
    count = 0
    for x in valid_data:
        nx = []
        for w in x:
            nx.append(w)
        numpy.array(nx,dtype="object",copy=True)
        nX.append(nx)
        valid_label[count] = int(valid_label_temp[count])
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
        test_label[count] = int(test_label_temp[count])
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
    train_dataset, valid_dataset, test_dataset = loadData(filePath + ".mat")


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
    file = open(filePath + "description.txt",'w')

    file.write("The network have " + str(channel_length) + "input nodes in the 1st layer.\n")
    file.write("The amount of samples in the dataset is " + str(sample_counts) +".\n")
    file.write("The number of classification classes is " + str(destinations) +".\n")
    file.write("The size of the first convolutional layer is " + str(layer1_input_length)+".\n")
    file.write('The number of convolutional filters is '+ str(number_of_con_filters)+ ",each kernel sizes "+ str(con_filter_length) + "X1.\n")
    file.write("There are "+str(number_of_full_layer_nodes)+" nodes in the fully connect layer.\n")

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
    file.write("The input to the first convolutional layer shapes 1X" + str(layer1_input_length) + "X1.\n"  )
    model.add(layer1)

    model.add(Activation('tanh'))

    #the max pooling layer after the first convolutional layer
    first_feature_map_size = (layer1_input_length - con_filter_length) / con_step_length + 1
    max_pooling_kernel_size = int(math.ceil(first_feature_map_size / max_pooling_feature_map_size))
    print("The max pooling kernel size is ", max_pooling_kernel_size)
    file.write("The max pooling kernel size is " + str(max_pooling_kernel_size) +".\n")
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
    sgd = SGD(lr = learning_ratio, decay = train_decay, momentum = 0.6, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    
    train_dataset_data = train_dataset[0].reshape(train_dataset[0].shape[0],1,train_dataset[0].shape[1],1)
 #   train_dataset_label = np_utils.to_categorical(train_dataset[1])
    file.close()
    #根据已有的代码去构建训练好的网络
    model.load_weights(filePath + 'Model.h5')
    test_dataset_data = test_dataset[0].reshape(test_dataset[0].shape[0],1,test_dataset[0].shape[1],1)
 #   test_dataset_label = np_utils.to_categorical(test_dataset[1])
    
    #根据已有的代码去构建训练好的网络
    model.load_weights(filePath + 'Model.h5')
    #拿到CNN全连接层提取到的特征
    train_data_for_svm = getMiddleOutPut(model,[train_dataset_data],5)
#    print("层号5，shape：",train_data_for_svm.shape)
    train_label_for_svm = train_dataset[1]
#    print("训练数据label的shape:",train_label_for_svm.shape)
    
    test_data_for_svm = getMiddleOutPut(model,[test_dataset_data],5)    
    test_dataset_label = test_dataset[1].astype(numpy.int) 
    test_label_for_svm = test_dataset[1]

    #下面这部分是把上面的CNN喂到SVM里面
#    pca = decomposition.RandomizedPCA(n_components = 100,whiten=True)
#    pca.fit(X_train)

#    X_train_pca = pca.transform(X_train)
#    X_test_pca = pca.transform(X_test)

    #原来gamma的值为0.0008
    kernel_1 = 'linear'
    kernel_2 = 'rbf'
    clf1 = svm.SVC(C=0.8, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00001, max_iter = -1)
    
    clf2 = svm.SVC(C=0.8, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00001, max_iter = -1)

    print("#####################################################")
    print("在CNN+SVM上的结果：")
    print("数据集",filePath)
    print("kernel为")
    print(kernel_2)
    print("开始训练")
    
    start_time = time.time()
    clf1.fit(train_data_for_svm, train_label_for_svm)
    end_time = time.time()
    print("训练用时:%f",(end_time-start_time))

    start_time = time.time()
    print("在测试集上的平均正确率为",clf1.score(test_data_for_svm, test_label_for_svm))
    end_time = time.time()
    print("测试用时：%f" % (end_time - start_time))
    #result = clf.predict(X_train)
    #correctRatio = np.mean(np.equal(result,Y_train))


    print("#####################################################")
    print("正在SVM上进行测试")
    
    print("数据集",filePath)
    print("kernel为")
    print(kernel_2)
    start_time= time.time()
    clf2.fit(train_dataset[0], train_dataset[1])
    end_time = time.time()
    print("训练用时:",end_time - start_time)

    start_time = time.time()
    print("在测试集上的平均正确率为",clf2.score(test_dataset[0],test_dataset[1]))
    end_time = time.time()
    print("测试用时：%f" % (end_time - start_time))
    #result = clf.predict(X_train)



    print("#####################################################")
    print("正在CNN上进行测试\n")

    classes = model.predict_classes(test_dataset_data)
    start_time = time.time()
    test_accuracy = numpy.mean(numpy.equal(test_dataset_label,classes))
    end_time = time.time()
    print("同一个测试集，在CNN上的正确率为：",test_accuracy)
    print("测试用时：%f" % (end_time - start_time))


#print("对测试数据集的预测结果为：",classes)
#    print("测试数据集中的真实结果为：",test_dataset_label)
#    print("一共得到测试结果",len(classes),"个，一共有",len(test_dataset_label),"个.")
#    count = 0
#    correctCount = 0

    
#    classes = model.predict_classes(test_dataset_data, verbose=1)
#    comparasion = zip(classes, test_dataset_label)
#    print(comparasion)
#    for x,y in range(classes)
#        if()

#    test_accuracy = numpy.mean(numpy.equal(test_dataset_label, classes))

#   print("SVM的分类精度为:",  test_accuracy)
    
#    return classes,test_dataset_label    

def network(path, con_step_length, max_pooling_feature_map_size):
    return temp_network(path, number_of_con_filters = 20, con_step_length = con_step_length, max_pooling_feature_map_size = max_pooling_feature_map_size, number_of_full_layer_nodes = 100, learning_ratio = 0.06, train_decay = 0.001)

if __name__ == '__main__':
    network("newKSC8020N4first",5,40)
#    temp_network("newPU1NWith2RestN.mat", number_of_con_filters = 20, con_step_length = 1, max_pooling_feature_map_size = 40, number_of_full_layer_nodes = 100, learning_ratio = 0.01, train_decay = 0.001)
