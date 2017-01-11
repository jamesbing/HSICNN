#!/usr/bin/env python
# coding=utf-8
import numpy as np
import lmdb
import caffe

import scipy.io as sio

#拷贝数据
Data = sio.loadmat('newKSC1N8.mat');
#分别存储训练集数据标签和测试机数据标签
print 'loading training data...'
XTrainData = Data['DataTr'];
YTrainLabel = Data['CIdTr'][0];
print '训练样本标签：',YTrainLabel;
XTestData = Data['DataTe'];
YTestLabel = Data['CIdTe'][0];
print '测试样本标签：',YTestLabel;
#这个N指的是训练和测试样本样本的数量，目前先手动指定
NTrain = 4145;
NTest = 1050;

map_size = XTrainData.nbytes * 10;

#创建训练集lmdb数据
env = lmdb.open('HSITrainlmdb',map_size = map_size);

#把数据取出来存进去
count = 0;
with env.begin(write=True) as txn:
    for i in range(NTrain):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1;
        datum.height = 1;
        datum.width = 1584;
#        print 'the shape of the hyperspectral data is: ', datum.channels,datum.height,datum.width;
        datum.data = XTrainData[i].tostring();

        datum.label = int(YTrainLabel[i]);
        str_id = '{:08}'.format(i);

        txn.put(str_id.encode('ascii'), datum.SerializeToString());
        count = count + 1;
print count, ' samples have been transformed in Train Data Set.';


map_size = XTestData.nbytes * 10;

#创建训练集lmdb数据
env = lmdb.open('HSITestlmdb',map_size = map_size);

#把数据取出来存进去
count = 0;
with env.begin(write=True) as txn:
    for i in range(NTest):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1;
        datum.height = 1;
        datum.width = 1584;
#        print 'the shape of the hyperspectral data is: ', datum.channels,datum.height,datum.width;
        datum.data = XTestData[i].tostring();

        datum.label = int(YTestLabel[i]);
        str_id = '{:08}'.format(i);

        txn.put(str_id.encode('ascii'), datum.SerializeToString());
        count = count + 1;
print count, ' samples have been transformed in Test Data Set.';
