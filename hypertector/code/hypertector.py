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
import hic
import scipy.io as sio

from sys import argv

experiment_path_prefix = '../experiments/'
data_prefix = '../data/'

def run_single(learning_ratio, network_type):

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
    if network_type == '1':
        HSICNN.run_network(file_name, neurons,neuronLayersCount, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)
    elif network_type == '2':
        hic.run_network(file_name, neurons,neuronLayersCount, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)
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
        cnnrf.run(file_name,trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, raws_size, lines_size)
    elif following_strategy == 3 != following_strategy != 5:
        #CNN+svm and CNN+RF
        
        print "now processing the cnn + svm joint framework..."
        cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)

        print "now processing the cnn + rf joint framework..."
        cnnrf.run(file_name, trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, raws_size, lines_size)

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

def run_batch(datasetName,strategies, neurons, neuronLayersCount, maxpoolings, fullLayers, batch_size, learning, train_decay, epoches, following_strategy, trees, learning_sample_ratios, dataset_format, experiment_times):
    for time_counts in range(experiment_times):
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
            cnnrf.run(file_name,trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, raws_size, lines_size)
        elif following_strategy == 3:
            print "now processing the cnn + svm joint framework..."
            cnnsvm.run(file_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
            print "now processing the cnn + rf joint framework..."
            cnnrf.run(file_name, trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, raws_size,lines_size,-1)
        file = open(file_name + "_experiment_description.txt", 'w')
        file.write("-------------Experiment Description-------------\n")
        file.write("Data set:" + file_name + "#\n")
        file.write("neighbor strategy:" + str(neighbors) + "#\n")
        file.write("Convolutional Neurons:" + str(neurons) + "#\n")
        file.write("Each Convolutional Neuron operate " + str(neuronLayersCount) + '#\n')
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

def predesigned_network(network_type):
    print "running mode:" + network_type
    prompt = ">"
    print "What kind of operation you want to run?"
    print "#1 Run a single experiment;" 
    print "#2 Run a batched experiment;"
    print "#3 analyse existing experimental results or doing further experiments on existing data"
    if_batch = int(raw_input(prompt))
    if if_batch == 1:
        run_single(0, network_type)
    elif if_batch == 2:
        print "#1: fixed CNN, different ratio; #2:..."
        run_type = int(raw_input(prompt))
        if run_type == 1:
            os.system('clear')
            print "============================================================================================"
            print "Enter a sery of numbers of the ratio of training samples, end with an 'e' or 'end',"
            print "if you want to use the default sequence 1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90, enter an 'a' or 'all':"
            ratios = []
            temp_ratio = raw_input(prompt)
            if temp_ratio == 'a' or temp_ratio == 'all':
                temp_ratio = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
            else:
                while temp_ratio != 'e' and temp_ratio != 'end':
                    ratios.append(int(temp_ratio))
                    temp_ratio = raw_input(prompt)
            #ratios = temp_ratio
            print ratios
#def run_batch(learning_ratio):
#            mix_model_svm_ratio = 0
#            file_name, neighbors = data_util.prepare(learning_ratio)
#            print "now gathering the parameters of the network..."
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

            os.system('clear')
            print "Now gathering network configuration parameters for prior proposed Cube CNN...."
            print "--------------------------------------------------------------------------------------------"
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
            #if network_type == '3':
            #    print "#4:compare the cube cnn with the new hic framework;"
            #    print "#5:TODO: train a mix assemble cnn-classifier model."
            #elif network_type == '1'
            print "#4:TODO: train a mix assemble cnn-classifier model."
            if network_type == '3':
                print "#5: run and compare the cube cnn with the new hic framework"
            following_strategy = int(raw_input(prompt))
            if network_type == '1' and following_strategy == 4:
                print "enter the ratio of svm classifier:"
                mix_model_svm_ratio = int(row_input(prompt))
            tress = 0
            if following_strategy == 2 or following_strategy == 3:
                print "enter the count of trees you want to set in Random Forest:"
                trees = int(raw_input(prompt))
            
            #if network_type == '3' and following_strategy == 4:
            #    print "Now gathering parameter for hic network:"

            
            print "How many individual experiments want to take?"
            experiment_times =  raw_input(prompt)

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
#                    for time_counts in range(int(experiment_times)):
                    file_name = run_batch(dataset_fixed,neighbor_strategy, neurons, neuronLayersCount, maxpoolings,fullLayers, batch_size_inner, learning_ratio, train_decay_inner, epoches, following_strategy, trees, ratios[temp_mark], 2, int(experiment_times))
                    #file_name = run_single(ratitemp_mark])
                    file.write(file_name + "\n")
                    fileCNNRFResultsPath = file_name + "_CNNRFdescription.txt"
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
                #file.close()
            resultFile.close()
            print "The results are stored in the file " + "BatchResults_" + time_stamp + ".txt"
            print "All folders contains the experiments are stored in the file " + "BatchExpsFixedCNN_" + time_stamp + ".txt"
    elif if_batch == 3:
        os.system('clear')
        analyse.analyse()

#def predesigned_network_HIC():
#    print "this network structure is designed based on the image type which proposed by Prof. Baigang on 2017,3,1."

def new_experiments():
    prompt = '>'
    
    #TODO choose the deep learning framework or backend.
    

    print "Choose The Backend or deep learning framework:"
    print "#1. Caffe  #2. Tensorflow  #3. Theano  #4. Both of them and compare."
    backend_type = raw_input(prompt)

    print "Want to: "
    print "#1 use the predesigned network, or"
    print "#2 do an experiment with a new framework"
    work_type = raw_input(prompt)
    if work_type == '1':
        predesigned_type = ''
        print "Want to:"
        print "#1 Operate CCS and CCR, or"
        print "#2 Operate HIC, or"
        print "#3 Operate both and compare them."
        predesigned_type = raw_input(prompt)
        while predesigned_type != '1' and predesigned_type != '2' and predesigned_type != '3':
            print "Entered wrong code, reenter please..."
            print "Want to:"
            print "#1 Operate CCS and CCR, or"
            print "#2 Operate HIC, or"
            print "#3 Operate both."
            predesigned_type = raw_input(prompt)
        predesigned_network(predesigned_type)
        #if predesigned_type == '1':
        #    predesigned_network_CCS_CCR()
        #elif predesigned_type == '2':
        #    predesigned_network_HIC()
        #elif predesigned_type == '3':
        #    predesigned_network_CCS_CCR()
        #    predesigned_network_HIC()
    elif work_type == '2':
        print "Choose the backend you want to use first, caffe, tensorflow or theano:"
        print "#1 caffe; #2 tensorflow; #3 theano"
        print "TODO"

def complete_experiments():
    folder_prompt = ">../experiments/"
    print "Select the experiments folder, if want to perform multi experiments, put their data as sub folders in the selected folder."
    file_path = raw_input(folder_prompt)
    true_file_path = experiment_path_prefix + file_path
    if os.path.exists(true_file_path) != True:
        print "Folder path \"" + true_file_path + "\" does not exist. Program termanited."
    else:
        #读取目录下所有的文件或者文件夹
        dirList = []
        perform_dir_list = ''
        complete_type = ''
        for temp_content in os.listdir(true_file_path):
            if os.path.isdir(os.path.join(true_file_path,temp_content)):
                print "目录" + temp_content
                dirList.append(os.path.join(true_file_path,temp_content))

        #判断是多组实验还是一个实验的逻辑
        if len(dirList) > 0:
            complete_type = 'multiple'
            perform_dir_list = dirList

        else:
            complete_type = 'one'
            perform_dir_list = true_file_path
        complete_implement(complete_type, perform_dir_list)

#complete_operate(operate_type = operate_type, folder_path = folder, trees = trees, neurons = neurons, neuronLayersCount = neuronLayersCount, maxpoolings = maxpoolings, fullLayers = fullLayers)
def complete_operate(operate_type, folder_path, trees, neurons, neuronLayersCount,maxpoolings, fullLayers):
    #TODO 添加增强健壮性的代码
    if os.path.isdir(folder_path):
        
        print "Processing " + folder_path
        #首先根据目录名字拿到数据集的文件名
        file_name_split = folder_path.split('/')
        file_or_folder_name = file_name_split[len(file_name_split) - 1]
        dataset_name_sub = file_or_folder_name.split('_')
        dataset_name = dataset_name_sub[0] + '_' + dataset_name_sub[1] + '_' + dataset_name_sub[2]
        
        #搜寻该文件夹中的网络配置文件
        #TODO:统一的网络配置文件格式
        #查询已有框架自动恢复网络结构的方式？？？
        #TODO:与下面的else配套
        #neurons = 0
        #neuronLayersCount = 0
        neighbors = int(dataset_name_sub[1]) + 1
        #maxpoolings = 0
        #fullLayers = 0
        raws = 0
        lines = 0
        batch_size = 100
        ratio = 0.001
        decay = 0.00001

        #这一块业务逻辑需要与调用该端代码的逻辑进行整合
        if os.path.exists(folder_path + '/networkconf.txt'):
            print "后续加上统一的网络配置文件记录之后就从这里面读取网络参数。"
        else:
            #TODO:下面代码是暂时的，而且这代码仅适用于CCS和CCR的工作，因此最终还是要靠networkconf.txt,以后改成没有这个配置文件就不让运行。
        #    print "网络参数配置文件不存在，请手动输入："
        #    print "Enter convolutional neurons in this network:"
         
        #    neurons = int(raw_input(prompt))
        #    print "Enter layers each convolutional neuron operates:"
        #    neuronLayersCount = int(raw_input(prompt))
        #    print "Enter maxpooling kernel size:"
        #    maxpoolings = int(raw_input(prompt))
        #    print "Enter fully layer neurons count:"
        #    fullLayers = int(raw_input(prompt))

            #rows lines 暂时先去数据集中找，以后这些也应该作为参数保存起来，直接load即可。
            LabelsMat = sio.loadmat(data_prefix + dataset_name_sub[0] + '/' + dataset_name_sub[0] + 'Gt.mat')
            key_label_name = LabelsMat.keys()
            label_key = ''
            for temp_key in key_label_name:
                if temp_key != '__version__' and temp_key != '__header__' and temp_key != '__globals__':
                    label_key = temp_key
                    break
            Labels = LabelsMat[label_key]
            raws = int(len(Labels))
            lines = int(len(Labels[0]))

        if operate_type == '1':
            print 'CNN+RF base on trained CNN model.'
            if len(trees.split('-')) > 0:
                small_tree_number = trees.split('-')[0]
                big_tree_number = trees.split('-')[1]
                exp_trees = range(int(small_tree_number),int(big_tree_number) + 1)
                cnnrf_acc_list = []
                rf_acc_list = []
                for trees_number in exp_trees:
                    print "Tree Number:" + str(trees_number)
                    cnnrf_acc, rf_acc = cnnrf.run(folder_path + "/" + dataset_name,trees_number, int(neurons), int(neuronLayersCount), int(neighbors), int(maxpoolings), int(fullLayers), raws, lines,test_cnn = -1)
                                       # cnnrf.run(file_name,trees, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, raws_size, lines_size)
                    cnnrf_acc_list.append(cnnrf_acc)
                    rf_acc_list.append(rf_acc)
                sio.savemat(folder_path + "/ALL_CNN_RF_EXP_RESULT.mat",{'cnnrf':cnnrf_acc_list, 'rf':rf_acc_list})
            else:
                    cnnrf.run(dataset_name,trees_number, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, 0, 0, 0, raws, lines)

        elif operate_type == '2':
            print 'Rebuild CNN_+ SVM model experiments...'
            cnnsvm.run(folder_path + "/" + dataset_name, neurons, neuronLayersCount, neighbors, maxpoolings, fullLayers, 1, 1, 1)

        elif operate_type == '3':
            print 'TODO'

        elif operate_type == '4':
            print 'TODO'

        elif operate_type == '5':
            print 'TODO'

        else:
            print 'Not under selection list, skip it.'
    else:
        print "not a folder, skipt it."


def complete_implement(type, dir):
    prompt = '>'
    print "Type:" + type + ", as follows:"
    for temp in dir:
        print temp
    print "Choose operate type:"
    print "1: CNN+RF based on trained CNN model;"
    print "2: CNN+SVM based on trained CNN model;"
    print "3: Fine tune trained CNN network;"
    print "4: Draw RGB graphs;"
    print "5: TODO..."
    #选择类型
    #1 已有CNN模型，进行CNN+RF的实验
    #2 已有CNN模型，进行CNN+SVM的实验
    #3 对CNN进行训练微调
    #4 没有RGB图像，补充RGB图像
    #5 其他暂时没想到的实验
    operate_type = raw_input(prompt)

    trees = ''
    if operate_type == '1':
        print "Enter the number of trees you want to set in RF, if you want it to be a series trees, use - to express it, for example: if you want to do a series trees experiments among 1 to 20, enter 1-20. no space or other char between numbers."
        trees = raw_input(prompt) 

    if type == 'one':
        #执行一个逻辑
        complete_operate(operate_type = operate_type, folder_path = dir, trees = trees, neurons = neurons, neuronLayersCount = neuronLayersCount, maxpoolings = maxpoolings, fullLayers = fullLayers)

    else:
        #执行一组的逻辑

        print "网络参数配置文件不存在，请手动输入："
        print "Enter convolutional neurons in this network:"
        neurons = int(raw_input(prompt))
        print "Enter layers each convolutional neuron operates:"
        neuronLayersCount = int(raw_input(prompt))
        print "Enter maxpooling kernel size:"
        maxpoolings = int(raw_input(prompt))
        print "Enter fully layer neurons count:"
        fullLayers = int(raw_input(prompt))

        for folder in dir:
            complete_operate(operate_type = operate_type, folder_path = folder, trees = trees, neurons = neurons, neuronLayersCount = neuronLayersCount, maxpoolings = maxpoolings, fullLayers = fullLayers)

if __name__ == "__main__":
    prompt = ">"
    os.system('clear')
    print "=======NK-Hypertector Hyperspectral Image Analysis And Experimental Tool======="
    print "Want to: "
    print "#1:Complete experiments on existing data;"
    print "#2:Do new experiments from scratch."
    if_new = raw_input(prompt)
    if if_new == '1':
        complete_experiments()
    elif if_new == '2':
        new_experiments()
