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
from sklearn.externals import joblib
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
    train_dataset, valid_dataset, test_dataset = loadData("/home/jiabing/HSICNNKSC/" + filePath + ".mat")

    file = open(filePath + "CNNSVMdescription.txt",'w')

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
    con_filter_length = int((math.ceil( (layer1_input_length /  con_step_length) / 9)) * con_step_length)

    destinations = numpy.max(test_dataset[1])
    
    #############################
    #Network Information Display#
    #############################
    #file = open(filePath + "description.txt",'w')

 #   file.write("The network have " + str(channel_length) + "input nodes in the 1st layer.\n")
 #   file.write("The amount of samples in the dataset is " + str(sample_counts) +".\n")
 #   file.write("The number of classification classes is " + str(destinations) +".\n")
 #   file.write("The size of the first convolutional layer is " + str(layer1_input_length)+".\n")
 #   file.write('The number of convolutional filters is '+ str(number_of_con_filters)+ ",each kernel sizes "+ str(con_filter_length) + "X1.\n")
 #   file.write("There are "+str(number_of_full_layer_nodes)+" nodes in the fully connect layer.\n")

 #   print("The network have ", channel_length, "input nodes in the 1st layer.")
 #   print("The amount of samples in the dataset is ", sample_counts)
 #   print("The number of classification classes is ", destinations)
 #   print("The size of the first convolutional layer is ", layer1_input_length)
 #   print('The number of convolutional filters is ', number_of_con_filters, ",each kernel sizes ", con_filter_length,"X1.")
 #  print("There are ",number_of_full_layer_nodes," nodes in the fully connect layer.")
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
#    file.close()
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

#	用rbf核
    clf1 = svm.SVC(C=1.0, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00000000000001, max_iter = -1)
    
    clf2 = svm.SVC(C=1.0, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00000000000001, max_iter = -1)
    #用linear
    clf3 = svm.SVC(C=0.8, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00001, max_iter = -1)    
    clf4 = svm.SVC(C=0.8, kernel = kernel_2,  gamma='auto', probability=True,
             tol = 0.00001, max_iter = -1)

    print("#####################################################")
    print("在CNN-SVM-RBF上的结果：")
    print("数据集",filePath)
    print("kernel为")
    print(kernel_2)
    print("开始训练")
    
    start_time = time.time()
    clf1.fit(train_data_for_svm, train_label_for_svm)
    end_time = time.time()
    train_time = end_time - start_time
    print("训练用时:",train_time)

    start_time = time.time()
    print("在测试集上的平均正确率为",clf1.score(test_data_for_svm, test_label_for_svm))
    end_time = time.time()
    test_time = end_time - start_time
    print("测试用时：%f" % test_time)
    #result = clf.predict(X_train)
    file.write("#########################################################################################################")
    file.write("The CNN-SVM joint use kernel " + kernel_2 + "\n")
    file.write("The SVM train time is " + str(train_time) +"\n")
    file.write("The testing time is " + str(test_time) + "\n")
    file.write("The correct ratio of CNN-SVM is " + str(clf1.score(test_data_for_svm,test_label_for_svm)))
    result = clf1.predict(test_data_for_svm)
    cnnsvmtraintime = str(train_time)
    cnnsvmtesttime = str(test_time)
    cnnsvmacc = str((clf1.score(test_data_for_svm,test_label_for_svm)))
    sio.savemat(filePath + "CNNSVMResult.mat",{'predict':result,'actual':test_label_for_svm})
    file.write("#########################################################################################################")
    joblib.dump(clf1,filePath + 'cnnsvmrbf.model')
		
    #result = clf.predict(X_train)
    #correctRatio = np.mean(np.equal(result,Y_train))

    print("#####################################################")
    print("正在SVM上进行测试")
    
    print("数据集",filePath)
    print("kernel为")
    print(kernel_2)
    print("开始训练")

    start_time= time.time()
    clf2.fit(train_dataset[0], train_dataset[1])
    end_time = time.time()
    train_time = end_time - start_time
    print("训练用时:",train_time)

    start_time = time.time()
    print("在测试集上的平均正确率为",clf2.score(test_dataset[0],test_dataset[1]))
    end_time = time.time()
    test_time = end_time - start_time
    print("测试用时：%f" % test_time)
    #result = clf.predict(X_train)
    file.write("#########################################################################################################")
    file.write("The SVM only use kernel " + kernel_2 + "\n")
    file.write("The SVM train time is " + str(train_time) +"\n")
    file.write("The testing time is " + str(test_time) + "\n")
    file.write("The correct ratio of CNN-SVM is " + str(clf2.score(test_dataset[0],test_dataset[1])))
    result = clf2.predict(test_dataset[0])
    svmtraintime = str(train_time)
    svmtesttime = str(test_time)
    svmacc = str(clf2.score(test_dataset[0],test_dataset[1]))
    sio.savemat(filePath + "SVMonlyResult.mat",{'predict':result,'actual':test_dataset[1]})
    file.write("#########################################################################################################")
    joblib.dump(clf2,filePath + 'svmrbf.model')

    print("#####################################################")
    print("正在CNN上进行测试\n")

    classes = model.predict_classes(test_dataset_data)
    start_time = time.time()
    test_accuracy = numpy.mean(numpy.equal(test_dataset_label,classes))
    end_time = time.time()
    print("同一个测试集，在CNN上的正确率为：",test_accuracy)
    print("测试用时：%f" % (end_time - start_time))
    file.write("#########################################################################################################")
    file.write("The CNN only\n")
    file.write("The testing time is " + str(end_time - start_time) + "\n")
    file.write("The correct ratio of CNN only is " + str(test_accuracy))
    sio.savemat(filePath + "CNNOnlyResult.mat",{'predict':classes,'actual':test_dataset_label})
    file.write("#########################################################################################################")
    cnntesttime = str(end_time - start_time)
    cnnacc = str(test_accuracy)
    return {'cnnsvmtraintime':cnnsvmtraintime,'cnnsvmtesttime':cnnsvmtesttime,'cnnsvmacc':cnnsvmacc, 'svmtraintime':svmtraintime,'svmtesttime':svmtesttime,'svmacc':svmacc,'cnntesttime':cnntesttime,'cnnacc':cnnacc}
    file.close

def network(path, con_step_length, max_pooling_feature_map_size):
    result =  temp_network(path, number_of_con_filters = 20, con_step_length = con_step_length, max_pooling_feature_map_size = max_pooling_feature_map_size, number_of_full_layer_nodes = 100, learning_ratio = 0.06, train_decay = 0.001)
    return result

def run_batch(flag):
    result1 = network("newKSC" + str(flag) + "N",1,40)
    result2 = network("newKSC" + str(flag) + "N4",5,40)
    result3 = network("newKSC" + str(flag) + "N8",9,40)
    return result1, result2, result3

if __name__ == '__main__':
    cnnsvmtraintime1 = 0.
    cnnsvmtesttime1 = 0.
    cnnsvmacc1 = 0.
    svmtraintime1 = 0.
    svmtesttime1 = 0.
    svmacc1 = 0.
    cnntesttime1 = 0.
    cnnacc1 = 0.

    cnnsvmtraintime4 = 0.
    cnnsvmtesttime4 = 0.
    cnnsvmacc4 = 0.
    svmtraintime4 = 0.
    svmtesttime4 = 0.
    svmacc4 = 0.
    cnntesttime4 = 0.
    cnnacc4 = 0.

    cnnsvmtraintime8 = 0.
    cnnsvmtesttime8 = 0.
    cnnsvmacc8 = 0.
    svmtraintime8 = 0.
    svmtesttime8 = 0.
    svmacc8 = 0.
    cnntesttime8 = 0.
    cnnacc8 = 0.
    
    file_all_result = open("KSCEXPResultTOTAL.txt",'w')

    for flag in range(1,31):
        if flag == 13:
            continue
        result = run_batch(flag)
        
        cnnsvmtraintime1 = cnnsvmtraintime1 + float(result[0]['cnnsvmtraintime'])
        cnnsvmtesttime1 = cnnsvmtesttime1 + float(result[0]['cnnsvmtesttime'])
        cnnsvmacc1 = cnnsvmacc1 + float(result[0]['cnnsvmacc'])
        svmtraintime1 = svmtraintime1 + float(result[0]['svmtraintime'])
        svmtesttime1 = svmtesttime1 + float(result[0]['svmtesttime'])
        svmacc1 = svmacc1 + float(result[0]['svmacc'])
        cnntesttime1 = cnntesttime1 + float(result[0]['cnntesttime'])
        cnnacc1 = cnnacc1 + float(result[0]['cnnacc'])

        cnnsvmtraintime4 = cnnsvmtraintime4 + float(result[1]['cnnsvmtraintime'])
        cnnsvmtesttime4 = cnnsvmtesttime4 + float(result[1]['cnnsvmtesttime'])
        cnnsvmacc4 = cnnsvmacc4 + float(result[1]['cnnsvmacc'])
        svmtraintime4 = svmtraintime4 + float(result[1]['svmtraintime'])
        svmtesttime4 = svmtesttime4 + float(result[1]['svmtesttime'])
        svmacc4 = svmacc4 + float(result[1]['svmacc'])
        cnntesttime4 = cnntesttime4 + float(result[1]['cnntesttime'])
        cnnacc4 = cnnacc4 + float(result[1]['cnnacc'])
        
        cnnsvmtraintime8 = cnnsvmtraintime8 + float(result[2]['cnnsvmtraintime'])
        cnnsvmtesttime8 = cnnsvmtesttime8 + float(result[2]['cnnsvmtesttime'])
        cnnsvmacc8 = cnnsvmacc8 + float(result[2]['cnnsvmacc'])
        svmtraintime8 = svmtraintime8 + float(result[2]['svmtraintime'])
        svmtesttime8 = svmtesttime8 + float(result[2]['svmtesttime'])
        svmacc8 = svmacc8 + float(result[2]['svmacc'])
        cnntesttime8 = cnntesttime8 + float(result[2]['cnntesttime'])
        cnnacc8 = cnnacc8 + float(result[2]['cnnacc'])
        file_all_result.write("|" +str(flag) + "|" + "1pixel" + "|" + result[0]['cnnsvmacc'] + "|" + result[0]['svmacc'] + "|" + result[0]['cnnacc'] + "|\n")

        file_all_result.write("|" +str(flag) + "|" + "4pixles" + "|" + result[1]['cnnsvmacc'] + "|" + result[1]['svmacc'] + "|" + result[1]['cnnacc'] + "|\n")
        
        file_all_result.write("|" +str(flag) + "|" + "8pixels" + "|" +result[2]['cnnsvmacc'] + "|" + result[2]['svmacc'] + "|" + result[2]['cnnacc'] + "|\n")
#        file_all_result.write(str(flag) + "|" + str(cnnsvmtraintime1) + "|" + str(cnnsvmtesttime1) + "|" + str(cnnsvmacc1) + "|" + str(svmtraintime1) + "|" + str(svmtesttime1) + "|" + str(svmacc1) + "|" + str(cnntesttime1) + "|" + str(cnnacc1) + "|\n")
#        file_all_result.write(str(flag) + "|" + str(cnnsvmtraintime4) + "|" + str(cnnsvmtesttime1) + "|" + str(cnnsvmacc1) + "|" + str(svmtraintime1) + "|" + str(svmtesttime1) + "|" + str(svmacc1) + "|" + str(cnntesttime1) + "|" + str(cnnacc1) + "|\n")
#        file_all_result.write(str(flag) + "|" + str(cnnsvmtraintime1) + "|" + str(cnnsvmtesttime1) + "|" + str(cnnsvmacc1) + "|" + str(svmtraintime1) + "|" + str(svmtesttime1) + "|" + str(svmacc1) + "|" + str(cnntesttime1) + "|" + str(cnnacc1) + "|\n")
    file = open("KSCEXPResultSUM.txt",'w')

    file.write("---------------------单像素-----------------------\n")

    file.write(str(cnnsvmtraintime1) + "\n")

    file.write(str(cnnsvmtesttime1) + "\n")

    file.write(str(cnnsvmacc1) + "\n")

    file.write(str(svmtraintime1) + "\n")

    file.write(str(svmtesttime1) + "\n")

    file.write(str(svmacc1) + "\n")

    file.write(str(cnntesttime1) + "\n")

    file.write(str(cnnacc1) + "\n")


    file.write("---------------------4近邻像素-----------------------\n")
    
    file.write(str(cnnsvmtraintime4) + "\n")

    file.write(str(cnnsvmtesttime4) + "\n")

    file.write(str(cnnsvmacc4) + "\n")

    file.write(str(svmtraintime4) + "\n")

    file.write(str(svmtesttime4) + "\n")

    file.write(str(svmacc4) + "\n")

    file.write(str(cnntesttime4) + "\n")
    
    file.write(str(cnnacc4) + "\n")


    file.write("---------------------8近邻像素-----------------------\n")

    file.write(str(cnnsvmtraintime8) + "\n")

    file.write(str(cnnsvmtesttime8) + "\n")

    file.write(str(cnnsvmacc8) + "\n")

    file.write(str(svmtraintime8) + "\n")

    file.write(str(svmtesttime8) + "\n")

    file.write(str(svmacc8) + "\n")

    file.write(str(cnntesttime8) + "\n")
    
    file.write(str(cnnacc8) + "\n")

    file.close

#    network("newKSC1N4",5,40)
#    network("newKSC1N4",5,40)
#    network("newKSC1N8",9,40)
#    temp_network("newPU1NWith2RestN.mat", number_of_con_filters = 20, con_step_length = 1, max_pooling_feature_map_size = 40, number_of_full_layer_nodes = 100, learning_ratio = 0.01, train_decay = 0.001)
