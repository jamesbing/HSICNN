#!/usr/bin/env python
# coding=utf-8
#@author jiabing leng
#@dajingjing
#@小冷今天去吃羊肉不带大静静
from __future__ import print_function
import numpy
import pandas as pd

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from keras.utils import np_utils

import scipy.io as sio
import random
import math
from keras import backend as K
from sklearn.externals import joblib
from sklearn import cross_validation,decomposition,metrics


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
def temp_network(filePath, trees, number_of_con_filters,conLayers,  con_step_length, max_pooling_feature_map_size, number_of_full_layer_nodes, learning_ratio, train_decay):
    #get the train data, train label, validate data, validate label, test data, test label
    train_dataset, valid_dataset, test_dataset = loadData(filePath + ".mat")

    file = open(filePath + "CNNRFdescription.txt",'w')

#    file.write("The network have " + str(channel_length) + "input nodes in the 1st layer.\n")
#    file.write("The amount of samples in the dataset is " + str(sample_counts) +".\n")
#    file.write("The number of classification classes is " + str(destinations) +".\n")
#    file.write("The size of the first convolutional layer is " + str(layer1_input_length)+".\n")
#    file.write('The number of convolutional filters is '+ str(number_of_con_filters)+ ",each kernel sizes "+ str(con_filter_length) + "X1.\n")
#    file.write("There are "+str(number_of_full_layer_nodes)+" nodes in the fully connect layer.\n")

    #the dimension of the input signal's chanel
    channel_length = train_dataset[0].shape[1]
    sample_counts = train_dataset[0].shape[0]

#    train_dataset, test_dataset = imdb.load_data()   
    #initialize parameters
    layer1_input_length = len(test_dataset[0][0])
    con_filter_length = int((math.ceil( (layer1_input_length /  con_step_length) / conLayers)) * con_step_length)
    destinations = numpy.max(test_dataset[1])
    model = Sequential()
    
    #the first convolutional layer
    layer1 = Convolution2D(number_of_con_filters,nb_row = con_filter_length, nb_col = 1,border_mode='valid', subsample=(1,1),dim_ordering='th', bias=True,input_shape=(1,layer1_input_length, 1))

    print("The input to the first convolutional layer shapes", (1,layer1_input_length,1))
    file.write("The input to the first convolutional layer shapes 1X" + str(layer1_input_length) + "X1.\n"  )
    model.add(layer1)

    model.add(Activation('tanh'))

    #the max pooling layer after the first convolutional layer
    first_feature_map_size = (layer1_input_length - con_filter_length) / con_step_length + 1
    #max_pooling_kernel_size = int(math.ceil(first_feature_map_size / max_pooling_feature_map_size))
    max_pooling_kernel_size = int(max_pooling_feature_map_size)
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
#    file.close()
    #根据已有的代码去构建训练好的网络
    model.load_weights(filePath + 'Model.h5')
    test_dataset_data = test_dataset[0].reshape(test_dataset[0].shape[0],1,test_dataset[0].shape[1],1)
 #   test_dataset_label = np_utils.to_categorical(test_dataset[1])
    
    #根据已有的代码去构建训练好的网络
    model.load_weights(filePath + 'Model.h5')
    #拿到CNN全连接层提取到的特征
    train_data_for_rf = getMiddleOutPut(model,[train_dataset_data],5)
#    print("层号5，shape：",train_data_for_rf.shape)
    train_label_for_rf = train_dataset[1]
#    print("训练数据label的shape:",train_label_for_rf.shape)
    
    test_data_for_rf = getMiddleOutPut(model,[test_dataset_data],5)    
    test_dataset_label = test_dataset[1].astype(numpy.int) 
    test_label_for_rf = test_dataset[1]

    #进行CNN+RF的综合实验
    #第一步：构造随机森林
    tree_counts = trees
    rf0 = RandomForestClassifier(n_estimators = tree_counts, oob_score = True, random_state = 10)
#
#y_predprob = gbml.predict_proba(train_data_for_rf)[:,1]
#    predict_ratio = metrics.roc_auc_score(train_label_for_rf, y_predprob)
#    print("the correct ratio on training dataset after the first attempt round is:" + str(predict_ratio))
    #第二步：确定森林中最佳的树的数量

#    print("the out of bag score after the first attempt round is: " + str(rf0.oob_score_))

    print("#####################################################")
    print("在CNN-RF上的结果：")
    print("数据集",filePath)
    print("树的数量：",tree_counts)
    print("开始训练")
    
    start_time = time.time()
    rf0.fit(train_data_for_rf, train_label_for_rf)
    end_time = time.time()
    train_time = end_time - start_time
    print("训练用时:",train_time)

    start_time = time.time()
    score = rf0.score(test_data_for_rf, test_label_for_rf)
    print("在测试集上的平均正确率为", score)
    end_time = time.time()
    test_time = end_time - start_time
    print("测试用时：%f" % test_time)
    #result = clf.predict(X_train)
    file.write("#########################################################################################################\n")
    file.write("The RF train time is " + str(train_time) +"\n")
    file.write("The testing time is " + str(test_time) + "\n")
    file.write("The tree number in this RF is " + str(tree_counts) + "\n")
    file.write("The correct ratio of CNN-RF is " + str(score) + "\n")
    result = rf0.predict(test_data_for_rf)
    cnnrftraintime = str(train_time)
    cnnrftesttime = str(test_time)
    cnnrfacc = str(score)
    sio.savemat(filePath + "CNNRFResult.mat",{'predict':result,'actual':test_label_for_rf})
    file.write("#########################################################################################################\n")
    joblib.dump(rf0,filePath + 'cnnrf.model')
		
    #result = clf.predict(X_train)
    #correctRatio = np.mean(np.equal(result,Y_train))
    
    #只采用RF的情况
    rf1 = RandomForestClassifier(n_estimators = tree_counts, oob_score = True, random_state = 10)
    print("#####################################################")
    print("采用原来的数据构建随机森林RF")
    
    print("数据集",filePath)
    print("开始训练")
    
    start_time= time.time()
    rf1.fit(train_dataset[0], train_dataset[1])
    end_time = time.time()
    train_time = end_time - start_time
    print("训练用时:",train_time)

    start_time = time.time()
    score_rf = rf1.score(test_dataset[0], test_dataset[1])
    print("在测试集上的平均正确率为",str(score_rf))
    end_time = time.time()
    test_time = end_time - start_time
    print("测试用时：%f" % test_time)
    #result = clf.predict(X_train)
    file.write("#########################################################################################################\n")
    file.write("The RF train time is " + str(train_time) +"\n")
    file.write("The testing time is " + str(test_time) + "\n")
    file.write("The correct ratio of RF only is " + str(score_rf) + "\n")
    result = rf1.predict(test_dataset[0])
    rftraintime = str(train_time)
    rftesttime = str(test_time)
    rfacc = str(score_rf)
    sio.savemat(filePath + "RFonlyResult.mat",{'predict':result,'actual':test_dataset[1]})
    file.write("#########################################################################################################\n")
    joblib.dump(rf1,filePath + 'rf.model')

    print("#####################################################")
    print("正在CNN上进行测试\n")

    classes = model.predict_classes(test_dataset_data)
    start_time = time.time()
    test_accuracy = numpy.mean(numpy.equal(test_dataset_label,classes))
    end_time = time.time()
    print("同一个测试集，在CNN上的正确率为：",test_accuracy)
    print("测试用时：%f" % (end_time - start_time))
    file.write("#########################################################################################################\n")
    file.write("The CNN only\n")
    file.write("The testing time is " + str(end_time - start_time) + "\n")
    file.write("The correct ratio of CNN only is " + str(test_accuracy) + "\n")
    sio.savemat(filePath + "CNNOnlyResult.mat",{'predict':classes,'actual':test_dataset_label})
    file.write("#########################################################################################################\n")
    cnntesttime = str(end_time - start_time)
    cnnacc = str(test_accuracy)
    return {'cnnrftraintime':cnnrftraintime,'cnnrftesttime':cnnrftesttime,'cnnrfacc':cnnrfacc, 'rftraintime':rftraintime,'rftesttime':rftesttime,'rfacc':rfacc,'cnntesttime':cnntesttime,'cnnacc':cnnacc}
    file.close

def network(file, trees, neurons, conLayers, convolutionalLayers, max_pooling_feature_map_size, full_layers_size, batch_size, ratio, decay):
    result =  temp_network(file, trees, number_of_con_filters = neurons,conLayers = conLayers, con_step_length = convolutionalLayers, max_pooling_feature_map_size = max_pooling_feature_map_size, number_of_full_layer_nodes = full_layers_size, learning_ratio = ratio, train_decay = decay)
    return result


def run(filename, trees, neurons, conLayers, neighbors, max_pooling_feature_map_size,full_layers_size,batch_size,ratio,decay):
    cnnrftraintime1 = 0.
    cnnrftesttime1 = 0.
    cnnrfacc1 = 0.
    rftraintime1 = 0.
    rftesttime1 = 0.
    rfacc1 = 0.
    cnntesttime1 = 0.
    cnnacc1 = 0.
    


    file = open(filename + "_CNNRF_EXPResultTOTAL.txt",'w')

    result = network(filename, trees, neurons,conLayers, neighbors,max_pooling_feature_map_size,full_layers_size,batch_size,ratio,decay)
        
    cnnrftraintime1 = cnnrftraintime1 + float(result['cnnrftraintime'])
    cnnrftesttime1 = cnnrftesttime1 + float(result['cnnrftesttime'])
    cnnrfacc1 = cnnrfacc1 + float(result['cnnrfacc'])
    rftraintime1 = rftraintime1 + float(result['rftraintime'])
    rftesttime1 = rftesttime1 + float(result['rftesttime'])
    rfacc1 = rfacc1 + float(result['rfacc'])
    cnntesttime1 = cnntesttime1 + float(result['cnntesttime'])
    cnnacc1 = cnnacc1 + float(result['cnnacc'])

    file.write("|" + filename + "results" + "|" + result['cnnrfacc'] + "|" + result['rfacc'] + "|" + result['cnnacc'] + "|\n")

    file.write("---------------------详细结果-----------------------\n")

    file.write(str(cnnrftraintime1) + "\n")

    file.write(str(cnnrftesttime1) + "\n")

    file.write(str(cnnrfacc1) + "\n")

    file.write(str(rftraintime1) + "\n")

    file.write(str(rftesttime1) + "\n")

    file.write(str(rfacc1) + "\n")

    file.write(str(cnntesttime1) + "\n")

    file.write(str(cnnacc1) + "\n")

    file.close

