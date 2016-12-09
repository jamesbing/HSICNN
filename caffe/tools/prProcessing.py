#!/usr/bin/env python
# coding=utf-8
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
    temp = raw_input(prompt)
    print temp
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
    
    lines = len(Labels)
    rows = len(Labels[0])
    indexRow = 0
    for row in Labels:
        indexLine = 0
        for line in row:
            label = line
            #store non-zero data
            if label != 0:
                #for test purpose printing...
                #print '[' + str(indexRow) + ',' + str(indexLine) + ']'
                temp_data = DataSet[indexRow,indexLine]
                if temp == 1:
                    data = temp_data
                else if temp == 4:
                    center_data = temp_data
                    if rowLine + 1


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
    





#DataSet




#processing code segment
print "enter the file folder path you want to transform..."
path = raw_input(prompt)

if os.path.exists(path_prefix + path) != True:
    print "you entered the wrong file folder path, please re-enter."
else:
    dataList = loadData(path)
    shuffledDataList = shuffling(dataList)
    print "please enter the ratio of training samples, eg. 80."

