#!/usr/bin/env python
# coding=utf-8
import os
import sys
context = '/home/para/caffe/'
sys.path.insert(0,context + '/python')
import caffe

import numpy as np
root = '/home/para/gitstore/HSICNN/vgg/'
deploy = root + 'train_test.prototxt'
caffe_model_root = root + 'experiment_results/'

def Test(model_sequence_number):
    caffe_model = caffe_model_root + model_sequence_number + '/_iter_10000.caffemodel'
    net = caffe.Net(deploy,caffe_model,caffe.TEST)

    out = net.forward()
    print out

if '__name__' == '__main__':
    Test(1)
