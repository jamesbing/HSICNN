#!/usr/bin/env python
# coding=utf-8
# dawang jinwan hele pidan doufutang. @ zhoujing @
# author @ jiabing leng @ nankai university @ tadakey@163.com

import time
import scipy.io as sio
import numpy as np
from random import shuffle
from sys import argv
from math import ceil
import lmdb
import sys
import os

#sys.path.insert(0,'/home/jiabing/caffe/python')

global neighbors 
#neighbors
prompt = '>'
context = '/home/para/caffe/'
datasetName = ''
#neighbors = 4
global train_ratio
neighbors = 4
path_prefix = '../data/'
sys.path.insert(0,context + '/python')
import caffe


def loadData(path, strategy):
    strategy = int(strategy)
    neighbors = strategy
    if strategy == 0:
        print 'please enter the neighbor pixels strategy, you can choose from 1,4 and 8.'
        neighbors = int(raw_input(prompt))
    #neighbors = neighbors
    print neighbors
#    while True:
#        if temp not in (1,4,8):
#            print 'you entered the wrong number, please re-enter.'
#            temp = raw_input(prompt)
#        else:
#            break

    
    dataset = path
    path = path_prefix + path
    print path
    
    #list all files under this folder
    #TODO should check if the files are correct under this folder to go preprocessing
    print "the folder contains following files:"
    for filename in os.listdir(path):
        print filename
    
    #load data and index file
    print 'validation dataset done, correct.'
    print 'loading data...'
    DataSetMat = sio.loadmat(path + '/' + dataset + 'Data.mat')
    LabelsMat = sio.loadmat(path + '/' + dataset + 'Gt.mat')
    key_data_name = DataSetMat.keys()
    key_label_name = LabelsMat.keys()
    # in case of the disorder of the content of data, should looking for the correct menu of the data index. for example,
    # the labels may be orginazed like['__version__', '__header__', 'ClsID', '__globals__'] rather than
    #['ClsID', all the other contents], so the code should not just use following index to fetch useable data.
    #DataSet = DataSetMat[key_data_name[0]]
    #Labels = LabelsMat[key_label_name[0]]

    data_key = ''
    label_key = ''
    for temp_key in key_data_name:
        if temp_key != '__version__' and temp_key != '__header__' and temp_key != '__globals__':
            data_key = temp_key
            break


    for temp_key in key_label_name:
        if temp_key != '__version__' and temp_key != '__header__' and temp_key != '__globals__':
            label_key = temp_key
            break

    DataSet = DataSetMat[data_key]
    Labels = LabelsMat[label_key]

    #DataSet = DataSetMat['DataSet']
    #Labels = LabelsMat['ClsID']
    maxClass = np.max(Labels)
    print 'there are ' + str(maxClass) + ' classes in dataset ' + str(dataset)
    print 'the spectral bands in this dataset is ' + str(len(DataSet[0][0]))
    
    #define many lists which number equals to maxClass,put it in a list
    #return shuffledDataList, neighbors, shuffledPositionList, rows, lines
    DataList = []
    PositionList = []
    shuffledDataList = []
    shuffledPositionList = []
    for mark in range(maxClass):
        DataList.append([])
        PositionList.append([])
        shuffledDataList.append([])
        shuffledPositionList.append([])

    #newDataset = np.array((Dataset.shape[0]))
#    newLabels = np.array()
#    for element in Labels.flat:
#        print element
    
    rows = len(Labels)
    lines = len(Labels[0])
    print 'size of the dataset: ' + str(rows), str(lines)

    indexRow = 0
    for indexRow in range(rows):
        indexLine = 0
        for indexLine in range(lines):
            label = Labels[indexRow,indexLine]
            #position = {'row':indexRow, 'line':indexLine}
            position = str(indexRow) + "|" + str(indexLine)
            # testing purpose:print position

            #store non-zero data
            if label != 0:
                #for test purpose printing...
                #print '[' + str(indexRow) + ',' + str(indexLine) + ']'
                data = DataSet[indexRow,indexLine]
                if  neighbors> 1:
                    center_data = data
                    #if indexRow + 1 < rows and rindexRow > 0 and indexLine + 1 < lines and indexLine > 0 and Labels[indexRow + 1,indexLine] !=0 and Labels[indexRow - 1,indexLine] != 0 and Labels[indexRow, indexLine + 1] !=0 and Labels[indexRow, indexLine -1] != 0:
                    #    data1 = DataSet[indexRow, indexLine - 1]
                    #    data2 = DataSet[indexRow, indexLine + 1]
                    #    data3 = DataSet[indexRow - 1, indexLine]
                    #    data4 = DataSet[indexRow + 1, indexLine]
                    #    data5 = DataSet[indexRow - 1, indexLine -1]
                    #    data6 = DataSet[indexRow - 1, indexLine + 1]
                    #    data7 = DataSet[indexRow + 1, indexLine - 1]
                    #    data8 = DataSet[indexRow + 1, indexLine + 1]
                    ####################################################################################################################
                    # fetching data around the target pixel according to following illustruction:
                    # 
                    #           data1      data2      data3
                    #           data4     center      data5
                    #           data6      data7      data8
                    ####################################################################################################################
                    data1 = []
                    data2 = []
                    data3 = []
                    data4 = []
                    data5 = []
                    data6 = []
                    data7 = []
                    data8 = []

                    # data1
                    if indexRow - 1 >= 0 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data1 = DataSet[indexRow - 1, indexLine - 1]
                    elif indexRow - 1 >= 0 and indexLine + 1 <= lines - 1 and Labels[indexRow - 1, indexLine + 1] > 0:
                        data1 = DataSet[indexRow - 1, indexLine + 1]
                    else:
                        data1 = center_data
                    
                    # data2
                    if indexRow - 1 >= 0 and Labels[indexRow - 1, indexLine] > 0:
                        data2 = DataSet[indexRow - 1, indexLine]
                    elif indexRow + 1 <= rows - 1 and Labels[indexRow + 1, indexLine] > 0:
                        data2 = DataSet[indexRow + 1, indexLine]
                    else:
                        data2 = center_data
                        
                    # data3
                    if indexRow - 1 >= 0 and indexLine + 1 <= lines - 1 and Labels[indexRow - 1, indexLine + 1] > 0:
                        data3 = DataSet[indexRow - 1, indexLine + 1]
                    elif indexRow - 1 >= 0 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data3 = DataSet[indexRow - 1, indexLine - 1]
                    else:
                        data3 = center_data
                        
                    # data4
                    if indexLine - 1 >= 0 and Labels[indexRow, indexLine - 1] > 0:
                        data4 = DataSet[indexRow, indexLine - 1]
                    elif indexLine + 1<= lines - 1 and Labels[indexRow, indexLine + 1] > 0:
                        data4 = DataSet[indexRow, indexLine + 1]
                    else:
                        data4 = center_data
                    
                    # data5
                    if indexLine + 1 <= lines - 1 and Labels[indexRow, indexLine + 1] > 0:
                        data5 = DataSet[indexRow, indexLine + 1]
                    elif indexLine - 1 >= 0 and Labels[indexRow, indexLine - 1] > 0:
                        data5 = DataSet[indexRow, indexLine - 1]
                    else:
                        data5 = center_data
                    
                    # data6
                    if indexRow + 1 <= rows - 1 and indexLine - 1 >= 0 and Labels[indexRow + 1, indexLine - 1] > 0:
                        data6 = DataSet[indexRow + 1, indexLine - 1]
                    elif indexRow + 1 <= rows - 1 and indexLine + 1 <= lines - 1 and Labels[indexRow + 1, indexLine + 1] > 0:
                        data6 = DataSet[indexRow + 1, indexLine + 1]
                    else:
                        data6 = center_data
                        
                        
                    # data7
                    if indexRow + 1 <= rows - 1 and Labels[indexRow + 1, indexLine] > 0:
                        data7 = DataSet[indexRow + 1, indexLine]
                    elif indexRow - 1 >= 0 and Labels[indexRow - 1, indexLine] > 0:
                        data7 = DataSet[indexRow - 1, indexLine]
                    else:
                        data7 = center_data
                        
                    # data8
                    if indexRow + 1 <= rows - 1 and indexLine + 1 <= lines - 1 and Labels[indexRow + 1, indexLine + 1] > 0:
                        data8 = DataSet[indexRow + 1, indexLine + 1]
                    elif indexRow + 1 <= rows - 1 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data8 = DataSet[indexRow + 1, indexLine - 1]
                    else:
                        data8 = center_data
                    
                    if neighbors == 4:
                        data_1 = np.append(data2, data4)
                        data_2 = np.append(data5, data7)
                        data_3 = np.append(data_1, data_2)
                        data  = np.append(data, data_3)
                        #data = data + data2 + data4 + data5 + data7
                    elif neighbors == 8:
                        #print data
                        #data = np.append(data, data1, data2, data3, data4, data5, data6, data7, data8)
#                        print "neighbor startegy is 8"
                        data_1 = np.append(data1, data2)
                        data_2 = np.append(data3, data4)
                        data_3 = np.append(data5, data6)
                        data_4 = np.append(data7, data8)
                        data_5 = np.append(data_1, data_2)
                        data_6 = np.append(data_3, data_4)
                        data_7 = np.append(data_5, data_6)
                        data = np.append(data, data_7)

                        #print data
                        #print 'data1' + str(data1) + 'data2 ' + str(data2) + 'data3' + str(data3)
                        #print 'data1 + data2:'
                        #print np.append(data1, data2)
                #    elif neighbors == 1:
                #        data = 

                DataList[label - 1].append(data)
                # the position string includes following informations:
                # row | line | class number.
                PositionList[label - 1].append(position + "|" + str(label - 1))
            indexLine = indexLine + 1

        indexRow = indexRow + 1
    
    print 'data loaded.'
    print 'spectral length now is: ' + str(len((DataList[0][0])))
    print 'neighbor strategy ' + str(neighbors)
    #DataList 用于存放数据和类别的二维List
    #neighbors 近邻策略
    #PositionList 用于存储位置信息的向量
    #rows 高光谱图像行数
    #lines 高光谱图像列数
    #进行shuffle  TODO后期将shuffle抽取出来，删掉之前的几个重复的shuffle，统一成一些函数。

    #shuffledDataList = []
    #shuffledPositionList = []
   
    shuffledDataList = DataList
    shuffledPositionList = PositionList
    print 'call data shuffling function...'
    
    for shuffleMarkCount in range(len(DataList)):

        tempDataList, tempPositionList = shuffling_tow_list(DataList[shuffleMarkCount], PositionList[shuffleMarkCount])
        print len(tempDataList)
        print len(tempPositionList)
        shuffledDataList[shuffleMarkCount] = tempDataList
        shuffledPositionList[shuffleMarkCount] = tempPositionList
    
    #print len(shuffledDataList)

    return shuffledDataList, neighbors, shuffledPositionList, rows, lines

def shuffling_tow_list(dataList, PositionList):
#    print PositionList[0]
    print 'shuffling data...'
#    shuffledA = []
#    shuffledB = []
#    for mark in range():
#        DataList.append([])
#        PositionList.append([])

    #if(len(listA) == len(positionList) != True):
    #    print 'The length of two lists does not match.'
    #    return 0
    listA = dataList
    listB = PositionList
    matched_length = len(listA)
    shuffleMark = range(matched_length)
    shuffledA = listA
    shuffledB = listB
    #print shuffleMark
    reloadMark = 0
    shuffle(shuffleMark)
    for tempCount in shuffleMark:
        #print len(listB[tempCount])
        shuffledA[reloadMark] = (listA[tempCount])
        shuffledB[reloadMark] = (listB[tempCount])
        reloadMark = reloadMark + 1

    print 'shuffled.'
#    print len(shuffledA)
    return shuffledA, shuffledB

def shuffling(dataList,ids, positionList):
    print 'shuffling data...'
    if (len(dataList) == len(positionList) and len(dataList) == len(ids)) != True:
        print 'The length of data list and position list does not match.'
        return 0
    shuffleMark = range(len(dataList))
    
    shuffledData = []
    shuffledIds = []
    shuffledPosition = []
    
    shuffle(shuffleMark)
    for tempCount in shuffleMark:
        shuffledData.append(dataList[tempCount])
        shuffledPosition.append(positionList[tempCount])
        shuffledIds.append(ids[tempCount])
#
#    for sub_list in dataList:
#        shuffle(sub_list)
    print 'shuffled.'
    return shuffledData, shuffledIds, shuffledPosition
    
def writeToLMDB(list, name, procedure):

    # prepare the data list
    #print list[0]
    new_big_list = []
    #add_count = 0
    classCount = 1
    for sub_list in list:
        #print 'samples number :' + str(len(sub_list))
        for sub_list_data in sub_list:
            print 'number of samples in this class ' + str(len(sub_list_data))
            for to_be_assemblied_data in sub_list_data:
                data_dict = {'label': classCount, 'data': to_be_assemblied_data}
                new_big_list.append(data_dict)
            classCount = classCount + 1
    # now the data format have been transformed into this:
    # new_big_list = [data_dicts....]
    # in which data_dict is {'label': a label, 'data': data value}
    # print new_big_list[0:20]
    #print 'shuffling data again among different classes....'
    #shuffle(new_big_list)
    #print new_big_list[0]['label']
    #print new_big_list[0]['data']
    print 'the number of spectral in this dataset is :' + str(len(new_big_list[0]['data']))

    map_size = sys.getsizeof(new_big_list) * 100000
    # prepare the lmdb format file
    print 'creating training lmdb ' + procedure + 'format dataset...'
    env = lmdb.open('HSI' + name + procedure + 'lmdb', map_size = map_size)
    #count = 0
    spectralBands = len(new_big_list[0]['data'])
    print 'this data set '+ name +' had spectral bands of ' + str(spectralBands)
    temp_i = 0
    countingMark = 0
    sampleCounts = range(len(new_big_list))
    shuffle(sampleCounts)
    with env.begin(write = True) as txn:
        for temp in sampleCounts:
            sample = new_big_list[temp]
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = 1
            datum.width = spectralBands
            # print sample
            datum.data = sample['data'].tostring()
            datum.label = int(sample['label'])
            str_id = '{:08}'.format(temp_i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
	    temp_i = temp_i + 1
            countingMark = countingMark + 1
	    #print '.'
    print 'Done.'
    print str(countingMark) + ' samples have successfully writed into lmdb format data file.'

def prepareMatList(list, positions):
    Data = []
    CId = []
    Positions = []
#    DataTe = []
#    CIdTe = []
    classCount = 1
    #positionMark_A = 0
    #PositionMark_B = 0
    #PositionMark_C = 0
    #print positions.shape
    #TODO: put these following two fors into one for.
    for sub_list in list:

        for sub_list_data in sub_list:

            print 'number of samples in number ' + str(classCount) + ' class ' + str(len(sub_list_data))
            for to_be_assemblied_data in sub_list_data:

                Data.append(to_be_assemblied_data) 
                CId.append(classCount)
                #Positions.append(positions[positionMark])
                #positionMark = positionMark + 1
                #PositionMark_C = PositionMark_C
                #Positions.append(positions[positionMark_A][PositionMark_B][PositionMark_C])
            classCount = classCount + 1
            #PositionMark_B = PositionMark_B + 1
        #positionMark_A = positionMark_A + 1
    for sub_positions in positions:
        #print str(len(sub_positions)) + '  '
        for sub_sub_positions in sub_positions:
            #print len(sub_sub_positions)
            #print str(len(sub_sub_positions))
            for actual_Position in sub_sub_positions:
            #    print str(len(actual_Positions))
                Positions.append(actual_Position)
    
    newData, newCId, newPositions = shuffling(Data, CId, Positions)
    
    return newData, newCId, newPositions


# write to .mat data format
def writeToMAT(trainList, testList,trainPositions, testPositions, datasetName, train_ratio, neighbors):
    DataTr, CIdTr, PositionsTr = prepareMatList(trainList, trainPositions)
    DataTe, CIdTe, PositionsTe = prepareMatList(testList, testPositions)
  
    ltime = time.localtime()
    time_stamp = str(ltime[0]) + "_" + str(ltime[1]) + "_" + str(ltime[2]) + "_" + str(ltime[3]) + "_" + str(ltime[4])

    folderPath = "../experiments/" + datasetName + '_' + str(neighbors) + '_' + str(train_ratio) + "_" + time_stamp + "/"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    realPath = folderPath + datasetName + "_" + str(neighbors) + "_" + str(train_ratio)

    sio.savemat(realPath + '.mat',{'DataTr':DataTr, 'CIdTr':CIdTr, 'PositionsTr':PositionsTr,  'DataTe':DataTe, 'CIdTe':CIdTe, 'PositionsTe':PositionsTe})
    return realPath, neighbors


def assembleData(list,positionList, datasetName, neighbors, learning_ratio, dataset_format):
    
    ratio = 0
    if learning_ratio == 0:
        print "please enter the ratio of training samples, eg. 80."
        ratio = int(raw_input(prompt))
        #train_ratio = ratio
    else:
        ratio = learning_ratio
    # prepare the lmdb format dataset
    # allocate the storage space for the dataset
    # TODO: check how to allocate space according to the specific dataset instead of use the following map_size directly.
    #map_size = list.nbytes * 0
    #create the lmdb data
    #envTrain = lmdb.open(datasetName + 'HSITrainlmdb', map_size = map_size)
    #envTest = lmdb.open(datasetName + 'HSITestlmdb', map_size = map_size)

    
    # split the dataset according to the ratio to caffe recognizable datasets
    positionMark = 0
    trainList = []
    testList = []
    trainPositions = []
    testPositions = []
    for mark in range(len(list)):
        trainList.append([])
        testList.append([])
        trainPositions.append([])
        testPositions.append([])
    print 'confirm the number of classes in this dataset is ' + str(len(list))
    trainingCount = 0
    testingCount = 0
    #for sub_list in list:
    positionMark = 0
    print '#########################ratioing############################'
    for dataList in list:
         positionNow = positionList[positionMark]
        #trainingNumer = ceil((len(dataList) * float(ratio) / 100.0)
        # print 'the number of samples in this class is :' + str(len(dataList))
         trainingNumber = int(ceil((len(dataList) * int(ratio)) / 100.0))
         testingNumber = int(len(dataList) - trainingNumber)
        # print 'the position of training list is from  0 to ' + str(trainingNumber)  + '.' 
         trainList[positionMark].append(dataList[0:trainingNumber])
         testList[positionMark].append(dataList[trainingNumber:len(dataList)])
         trainPositions[positionMark].append(positionNow[0:trainingNumber])
         testPositions[positionMark].append(positionNow[trainingNumber:len(dataList)])
         trainingCount = trainingCount + trainingNumber
         print '.............................................................'
         print 'class ' + str(positionMark)
         print 'train samples\' count:' + str(trainingNumber)
         testingCount = testingCount + testingNumber
         print 'test samples\' count:' + str(testingNumber)
         print str(len(dataList)) + '.'
         positionMark = positionMark + 1
    print '---------------------------------------------------------------'
    print 'data splited in to different datasets:'
    print 'there are ' + str(trainingCount) + ' training samples and '
    print 'there are ' + str(testingCount) + ' testing samples.'
    print 'writing dataset...'

    data_format = 0
    #dataset_format = int(dataset_format)
    print dataset_format
    if dataset_format == "" and dataset_format != 1 and dataset_format != 2:
        print "choose the data format, enter 1 for lmdb or enter 2 for mat"
        data_format = int(raw_input(prompt))
    elif dataset_format == 1 or dataset_format == 2:
        data_format = dataset_format
    if data_format == 1:
        # write the splited data into lmdb format files
        writeToLMDB(trainList, datasetName, 'training')
        writeToLMDB(testList, datasetName, 'testing')
    elif data_format == 2:
        return writeToMAT(trainList, testList, trainPositions, testPositions, datasetName, ratio, neighbors)

#def assembleData(list, datasetName):
#    print "choose the data format, enter 1 for lmdb or enter 2 for mat"
#    data_format = int(raw_input(prompt))
#    if data_format == 1:
#        assembleLMDB(list, datasetName)
#    elif:
#        assembleMAT(list, datasetName)

#processing code segment
def prepare(learning_ratio, data_set, neighbors, dataset_format):
    #print "want to #1:construct a new dataset or #2:use existing dataset?"
    #if_new = int(raw_input(prompt))
    if_new = 1
    # judge if the dataset is exists. to determain if the code will use the existing dataset and the exsiting experiment results.
    if if_new == 1:
        path = data_set
        if data_set == "NONE":
            print "enter the file folder path you want to transform..."    
            path = raw_input(prompt)
        
        if os.path.exists(path_prefix + path) != True:
            print "you entered the wrong file folder path, please re-enter."
        else:
            dataList, inner_neighbors, positionList, rows, lines = loadData(path, neighbors)
            #for testing purpose:print positionList
            #shuffledDataList, shuffledPositionList = shuffling(dataList, positionList)
            print len(dataList[0])
            print len(positionList[0])
            realPath, wrong_neighbor = assembleData(dataList, positionList, path, inner_neighbors, learning_ratio, dataset_format)
            #realPath = path + '_' + str(neighbors) + '_' + str(train_ratio)i
            print "the dataset is stored in " + realPath + ".mat"
            print inner_neighbors
            return realPath, inner_neighbors, rows, lines
    elif if_new == 2:
        print "enter the existing dataset path:"
        realPath = raw_input(prompt)
        #TODO后期要根据路径名去判断数据集的信息，并且赋给neighbors 变量，暂先用8固定
        neighbors = 8
        realPath = "../experiments/" + realPath + "/" +realPath
        return realPath, neighbors, raws, lines

if __name__ == '__main__':
    prepare(0,'KSC', 8, 2)
