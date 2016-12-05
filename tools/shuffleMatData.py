#!/usr/bin/env python
# coding=utf-8
import scipy.io as sio
import numpy as np
from random import shuffle

dataset = sio.loadmat('newKSC3N8.mat')
traindata = dataset['DataTr']
trainlabel = dataset['CIdTr']
testdata = dataset['DataTe']
testlabel = dataset['CIdTe']
#打印信息
print '读取数据成功，准备打乱数据....'

#初始化打乱后的数据
newTrainData = np.array((traindata.shape[0],traindata.shape[1]),dtype=np.float64)
newTrainLabel = np.array((trainlabel.shape[0],trainlabel.shape[1]),dtype=np.uint8)
newTestData = np.array((testdata.shape[0],testdata.shape[1]),dtype=np.float64)
newTestLabel = np.array((testlabel.shape[0],testlabel.shape[1]),dtype=np.uint8)

#用来打乱数据集以及对应的标签
def shuffleDataAndLabel(data, label):
    mark = [[x] for x in range(data.shape[0])]
    shuffle(mark)
    tempCount = 0
    newData = np.zeros((data.shape[0],data.shape[1]),dtype=np.float64)
    newLabel = np.zeros((label.shape[0],label.shape[1]),dtype=np.uint8)
    for temp in mark:
        i = temp[0]
        newData[tempCount] = data[i]
        newLabel[0][tempCount] = label[0][i]
        tempCount = tempCount + 1

    return newData, newLabel


newTrainData,newTrainLabel = shuffleDataAndLabel(traindata,trainlabel)
newTestData,newTestLabel = shuffleDataAndLabel(testdata,testlabel)

#将得到的新数组存储到matlab文件中
sava_file_name = 'newKSC.mat'

print '打乱前训练集数据标签序列:'
print trainlabel
print '打乱后训练集数据标签序列:'
print newTrainLabel
print '打乱完成。'

sio.savemat(sava_file_name,{'DataTr':newTrainData,'CIdTr':newTrainLabel,'DataTe':newTestData,'CIdTe':newTestLabel})
