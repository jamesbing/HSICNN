#!/usr/bin/env python
# coding=utf-8
# O(∩_∩)O @ zhoujing @
# author @ jiabing leng @ nankai university @ tadakey@163.com
import HSICNN
import data_util
import cnnsvm
import cnnrf
import time
import analyse
import os

from sys import argv

def run_single(learning_ratio):

    prompt = '>'
    #mix_model_svm_ratio是为了以后采用组合混合模型的时候，保存一个svm在所有模型中的占比。以后根据需求进行扩充。。。TODO
    mix_model_svm_ratio = 0
    file_name, neighbors, raws_size, lines_size = data_util.prepare(learning_ratio,"NONE", 0, 2)

    print "now constructing the network..."
    #print "enter the layers each convolutional kernel covers: "
    #neighbors = int(raw_input(prompt))
    neighbors = neighbors + 1

    print "the neighbors strategy is: " + str(neighbors)
    print "enter the number of convolutional neurons:"
    neurons = int(raw_input(prompt))
    print "enter the number of layers you want the CNN to operate convolutional operation:"
    neuronLayersCount = int(raw_input(prompt))
    print "enter the kernel size of the maxpooling layer:"
    maxpoolings = int(raw_input(prompt))
    print "enter the number of full layers\' neurons, default is 100:"
    fullLayers = int(raw_input(prompt))
    #if tempfullLayers > 1:
    #    fullLayers = tempfullLayers
    print "enter the batch size for bsgd:"
    batch_size = int(raw_input(prompt))
    print "enter the learning ratio:"
    learning = float(raw_input(prompt))
    print "enter the train decay:"
    train_decay = float(raw_input(prompt))
    print "enter the epoches you want the network to be trained:"
    epoches = int(raw_input(prompt))

    print "now choose the following strategy after the cnn network been trained:"
    print "#1:train a cnn-svm joint framework;"
    print "#2:train a cnn-rf joint framework;"
    print "#3:train both cnn-svm and cnn-rf joint frameworks;"
    print "#4:TODO: train a mix assemble cnn-classifier model."
    print "#5: train a CNN model only."
    following_strategy = int(raw_input(prompt))
    if following_strategy == 4:
        print "enter the ratio of svm classifier:"
        mix_model_svm_ratio = int(row_input(prompt))

    tress = 0
    if following_strategy == 2 or following_strategy == 3:
        print "enter the count of trees you want to set in Random Forest:"
        trees = int(raw_input(prompt))

    print "starting ..."
    HSICNN.run_network(file_name, neurons,neuronLayersCount, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)

    print "the training of the network have done."

    if following_strategy == 1 and following_strategy != 5:
        #CNN + SVM
        print "now processing the cnn + svm joint framework..."
        cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
    elif following_strategy == 2 and following_strategy != 5:
        #CNN + rfind
        print "now processing the cnn + rf joint framework..."
    #    print "enter the count of trees you want to set in Random Forest:"
    #    trees = int(raw_input(prompt))
        cnnrf.run(file_name,trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay, raws_size, lines_size)
    elif following_strategy == 3 != following_strategy != 5:
        #CNN+svm and CNN+RF
        
        print "now processing the cnn + svm joint framework..."
        cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)

        print "now processing the cnn + rf joint framework..."
        cnnrf.run(file_name, trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay, raws_size, lines_size)

    file = open(file_name + "_experiment_description.txt", 'w')
    file.write("-------------Experiment Description-------------\n")
    file.write("Data set:" + file_name + "#\n")
    file.write("neighbor strategy:" + str(neighbors) + "#\n")
    file.write("Convolutional Neurons:" + str(neurons) + "#\n")
    file.write("Each convolutional neuron operates " + str(neuronLayersCount))
    file.write("Max Polling Kernel Size:" + str(maxpoolings) + "#\n")
    file.write("Full Layer Neuron number:" + str(fullLayers) + "#\n")
    file.write("Batch size of SGD training:" + str(batch_size) + "#\n")
    file.write("Training epoches of deep CNN:" + str(epoches) + "#\n")
    file.write("Learning ratio:" + str(learning) + "#\n")
    file.write("Train decay:" + str(train_decay) +"#\n")
    if following_strategy == 2 or following_strategy == 3:
        file.write("Number of trees in random forest: " + str(trees) + "#\n")
    file.write("===============================================\n")
    file.close()
    return file_name

def run_batch(datasetName,strategies, neurons, neuronLayersCount, maxpoolings, fullLayers, batch_size, learning, train_decay, epoches, following_strategy, trees, learning_sample_ratios, dataset_format):
    mix_model_svm_ratio = 0
    #print strategies
    file_name, neighbors, raws_size, lines_size = data_util.prepare(learning_sample_ratios, datasetName, int(strategies), 2)
    neighbors = neighbors + 1
    print "the neighbors strategy is: " + str(neighbors)
    print "starting ..."
    HSICNN.run_network(file_name, neurons,neuronLayersCount, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)
    print "the training of the network have done."
    if following_strategy == 1:
        print "now processing the cnn + svm joint framework..."
        cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
    elif following_strategy == 2:
        print "now processing the cnn + rf joint framework..."
        cnnrf.run(file_name,trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay, raws_size, lines_size)
    elif following_strategy == 3:
        print "now processing the cnn + svm joint framework..."
        cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
        print "now processing the cnn + rf joint framework..."
        cnnrf.run(file_name, trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay, raws_size,lines_size)
    file = open(file_name + "_experiment_description.txt", 'w')
    file.write("-------------Experiment Description-------------\n")
    file.write("Data set:" + file_name + "#\n")
    file.write("neighbor strategy:" + str(neighbors) + "#\n")
    file.write("Convolutional Neurons:" + str(neurons) + "#\n")
    file.write("Each Convolutional Neuron operate " + str(neuronLayersCount))
    file.write("Max Polling Kernel Size:" + str(maxpoolings) + "#\n")
    file.write("Full Layer Neuron number:" + str(fullLayers) + "#\n")
    file.write("Batch size of SGD training:" + str(batch_size) + "#\n")
    file.write("Training epoches of deep CNN:" + str(epoches) + "#\n")
    file.write("Learning ratio:" + str(learning) + "#\n")
    file.write("Train decay:" + str(train_decay) +"#\n")
    if following_strategy == 2 or following_strategy == 3:
        file.write("Number of trees in random forest: " + str(trees) + "#\n")
    file.write("===============================================\n")
    file.close()
    return file_name

if __name__ == '__main__':
    prompt = ">"
    print "What kind of operation you want to run?"
    print "#1 Run a single experiment; #2 Run a batched experiment; #3 analyse existing experimental results or doing further experiments on existing data"
    if_batch = int(raw_input(prompt))
    if if_batch == 1:
        run_single(0)
    elif if_batch == 2:
        print "#1: fixed CNN, different ratio; #2:..."
        run_type = int(raw_input(prompt))
        if run_type == 1:
            print "enter a sery of numbers of the ratio of training samples, end with an 'e' or 'end', if you want to use the default sequence 1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90, enter an 'a' or 'all':"
            ratios = []
            temp_ratio = raw_input(prompt)
            if temp_ratio == 'a' or temp_ratio == 'all':
                temp_ratio = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
            else:
                while temp_ratio != 'e' and temp_ratio != 'end':
                    ratios.append(int(temp_ratio))
                    temp_ratio = raw_input(prompt)
            ratios = temp_ratio
#def run_batch(learning_ratio):
#            mix_model_svm_ratio = 0
#            file_name, neighbors = data_util.prepare(learning_ratio)
            print "now gathering the parameters of the network..."
#            neighbors = neighbors + 1
#            print "the neighbors strategy is: " + str(neighbors)
            print "enter the dataset name:"
            dataset_fixed = raw_input(prompt)
            print "enter the neighbor strategy, choose from 1, 4, or 8, end with an 'e' or 'end'. if you want to run on all the strategies, enter an 'a' or 'all' for all 1,4,8 strategies."
            temp_strategies_list = []
            temp_strategy_input = raw_input(prompt)
            if temp_strategy_input == 'a' or temp_strategy_input == 'all':
                temp_strategies_list = [1,4,8]
            else:
                while temp_strategy_input != 'e' and temp_strategy_input != 'end':
                    temp_strategies_list.append(int(temp_strategy_input))
                    temp_strategy_input = raw_input(prompt)
            #strategy_fixed = raw_input(prompt)

            print "enter the number of convolutional neurons:"
            neurons = int(raw_input(prompt))
            print "enter the number of layers you want the CNN to operate convolutional operation:"
            neuronLayersCount = int(raw_input(prompt))
            print "enter the kernel size of the maxpooling layer:"
            maxpoolings = int(raw_input(prompt))
            print "enter the number of full layers\' neurons, default is 100:"
            fullLayers = int(raw_input(prompt))
            print "enter the batch size for bsgd:"
            batch_size = int(raw_input(prompt))
            print "enter the learning ratio:"
            learning = float(raw_input(prompt))
            print "enter the train decay:"
            train_decay = float(raw_input(prompt))
            print "enter the epoches you want the network to be trained:"
            epoches = int(raw_input(prompt))
            print "now choose the following strategy after the cnn network been trained:"
            print "#1:train a cnn-svm joint framework;"
            print "#2:train a cnn-rf joint framework;"
            print "#3:train both cnn-svm and cnn-rf joint frameworks;"
            print "#4:TODO: train a mix assemble cnn-classifier model."
            following_strategy = int(raw_input(prompt))
            if following_strategy == 4:
                print "enter the ratio of svm classifier:"
                mix_model_svm_ratio = int(row_input(prompt))
            tress = 0
            if following_strategy == 2 or following_strategy == 3:
                print "enter the count of trees you want to set in Random Forest:"
                trees = int(raw_input(prompt))

            ltime = time.localtime()
            time_stamp = str(ltime[0]) + "#" + str(ltime[1]) + "#" + str(ltime[2]) + "#" + str(ltime[3]) + "#" + str(ltime[4])

            file = open("../experiments/BatchExpsFixedCNN_" + time_stamp + ".txt", 'w')
            resultFile = open("../experiments/BatchResults_" + time_stamp + ".txt", 'w')
            file.write("======== Experimental Folders ==========\n")
            resultFile.write("=============== Batch Exprimental Results ===============\n")
            resultFile.write("=========================================================\n")
            
            #strategiesList = []
            #if str(strategy_fixed) == 'a' or strategy_fixed == 'all':
            #    strategiesList = [1,4,8]
            #else:
            #    strategiesList = [int(strategy_fixed)]
            # 
            strategiesList = temp_strategies_list
            for neighbor_strategy_mark in range(len(strategiesList)):
                neighbor_strategy = strategiesList[neighbor_strategy_mark]
                print "now is running on strategy " + str(neighbor_strategy)
                file.write("~~~~~~~~~~~~~~~ Neighbors Strategies:" + str(neighbor_strategy) +" ~~~~~~~~~~~~~~~\n")
                for temp_mark in range(len(ratios)):
                    learning_ratio = 0
                    train_decay_inner = 0
                    batch_size_inner = 0
                    if ratios[temp_mark] < 10:
                        learning_ratio = learning / 10
                        train_decay_inner = train_decay / 10
                        batch_size_inner = batch_size / 10
                    #elif ratios[temp_mark] < 5:
                    #    learning_ratio = learning / 100
                    #    train_decay_inner = train_decay / 100
                    #    batch_size_inner = batch_size / 100
                    else:
                        learning_ratio = learning
                        train_decay_inner = train_decay
                        batch_size_inner = batch_size
    
                    #set the full layers nodes to satisfy the change of neighbors strategies.
                    #TODO: need to check if this makes sense
                    #actual_full_layers = 0
                    #if neighbor_strategy == 4:
                    #    actual_full_layers = fullLayers / 2
                    #elif neighbor_strategy == 1:
                    #    actual_full_layers = fullLayers / 4

                    file_name = run_batch(dataset_fixed,neighbor_strategy, neurons, neuronLayersCount, maxpoolings,fullLayers, batch_size_inner, learning_ratio, train_decay_inner, epoches, following_strategy, trees, ratios[temp_mark], 2)
                    #file_name = run_single(ratios[temp_mark])
                    file.write(file_name + "\n")
                    fileCNNRFResultsPath = file_name + "CNNRFdescription.txt"
                    if following_strategy == 3:
                        fileCNNSVMResultsPath = file_name + "CNNSVMdescription.txt"
                    resultFile.write("=========================================================\n")
                    resultFile.write(file_name + "\n")
                    inputFileRF = open(fileCNNRFResultsPath, "r")
                    if following_strategy == 3:
                        inputFileSVM = open(fileCNNSVMResultsPath, "r")
                    allLinesRF = inputFileRF.readlines()
                    if following_strategy == 3:
                        allLinesSVM = inputFileSVM.readlines()
                    resultFile.write("CNN-RF Results:\n")
                    for eachLine in allLinesRF:
                        resultFile.write(eachLine)
                    resultFile.write("-----------------------------------------\n")
                    if following_strategy == 3:
                        resultFile.write("CNN-SVM Results:\n")
                        for eachLine in allLinesSVM:
                            resultFile.write(eachLine)
                        inputFileRF.close()
                        inputFileSVM.close()
                    resultFile.write("##################################################\n")
            file.close()
            resultFile.close()
            print "The results are stored in the file " + "BatchResults_" + time_stamp + ".txt"
            print "All folders contains the experiments are stored in the file " + "BatchExpsFixedCNN_" + time_stamp + ".txt"
    elif if_batch == 3:
        os.system('clear')
        analyse.analyse()
