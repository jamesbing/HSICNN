#!/usr/bin/env python
# coding=utf-8
# author @ jiabing leng @ nankai university @ tadakey@163.com

import scipy.io as sio
import numpy as np
from random import shuffle
from sys import argv
import lmdb
import sys
import os

#sys.path.insert(0,'/home/jiabing/caffe/python')

prompt = '>'
context = '/home/jiabing/caffe/'
path_prefix = context + '/examples/HSI/datasets/'
sys.path.insert(0,context + '/python')

def loadData(path):
    print 'please enter the neighbor pixels strategy, you can choose from 1,4 and 8.'
    neighbors = raw_input(prompt)
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
    DataSet = DataSetMat['DataSet']
    Labels = LabelsMat['ClsID']
    maxClass = np.max(Labels)
    print 'there are ' + str(maxClass) + ' classes in dataset ' + str(dataset)
    
    #define many lists which number equals to maxClass,put it in a list
    DataList = []
    for mark in range(maxClass):
        DataList.append([])

    #newDataset = np.array((Dataset.shape[0]))
#    newLabels = np.array()
#    for element in Labels.flat:
#        print element
    
    rows = len(Labels)
    lines = len(Labels[0])
    print 'size of the dataset: ' + str(rows), str(lines)

    indexRow = 0
    for row in Labels:
        indexLine = 0
        for line in row:
            label = line
            #store non-zero data
            if label != 0:
                #for test purpose printing...
                #print '[' + str(indexRow) + ',' + str(indexLine) + ']'
                data = DataSet[indexRow,indexLine]
                if temp > 1:
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
                        data = data + data2 + data4 + data5 + data7
                    elif neighbors == 8:
                        data = data + data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8        
                    
                DataList[label - 1].append(data)
            indexLine = indexLine + 1

        indexRow = indexRow + 1
    
    print 'data loaded.'

    return DataList


def shuffling(dataList):
    print 'shuffling data...'
    for sub_list in dataList:
        shuffle(sub_list)
    print 'shuffled.'
    

def splitDataSet(list):
    print "please enter the ratio of training samples, eg. 80."
    ratio = raw_input(prompt)



#processing code segment
print "enter the file folder path you want to transform..."
path = raw_input(prompt)

if os.path.exists(path_prefix + path) != True:
    print "you entered the wrong file folder path, please re-enter."
else:
    dataList = loadData(path)
    shuffledDataList = shuffling(dataList)
    splitData(shuffledDataList)

