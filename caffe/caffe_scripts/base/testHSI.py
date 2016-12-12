#!/usr/bin/env python
# coding=utf-8
# this script is used to test the trained HSI CNN network.
# @james leng

import numpy as np
import sys
import lmdb
from sklearn.cross_validation import StratifiedShuffledSplit
import pandas as pd

caffe_root = '../../'
sys.path.insert(0,caffe_root + 'python')

import caffe

caffe.set_mode_gpu()

model_def = '/home/jiabing/caffe/examples/HSI/hsi_train_test.prototxt'
model_weights = '/home/jiabing/caffe/examples/HSI/hsi_iter_121500.caffemodel'
net = caffe.Net(model_def, 
               model_weights,
               caffe.TEST)

def load_data_

labels_file = '/home/jiabing/caffe/examples/HSI/hs'


