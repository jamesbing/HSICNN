#!/usr/bin/env python
# coding=utf-8

import numpy as np
import lmdb
import caffe
import scipy.io as sio

#filepath表示高光谱图图像的存储位置
filepath = 'newKSC1N8.mat'
#N表示样本的数量
N = 1050
#spectralLength表示波段长度
spectralLength = 1584
#path表示的是要生成的lmdb文件的名字
path = 'HSITestlmdb'

#装载高光谱图像数据,训练集
TotalData = sio.loadmat(filepath)
data = TotalData['DataTe'];
#print data
labels = TotalData['CIdTe'][0];
#print labels
print 'Loading data, file:', path, ',total sample number:', N, '...'

#for temp in labels:
#    print temp


#初始化
X = np.zeros((N, 1, 1, spectralLength), dtype=np.float64)
y = np.zeros(N, dtype=np.int16)
#print type(X)
#将高光谱数据拷贝到X y中
temp = 0
for tempData in data:
    X[temp] = tempData
    y[temp] = labels[temp]
    temp = temp + 1
print X
# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

map_size = X.nbytes * 100

env = lmdb.open(path, map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

print 'Done.'
