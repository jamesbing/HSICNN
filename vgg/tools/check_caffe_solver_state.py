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
    caffe_model = caffe_model_root + str(model_sequence_number) + '/_iter_10000.caffemodel'
    net = caffe.Net(deploy,caffe_model,caffe.TEST)

    out = net.forward()
    print "准确精度："
    print out['accuracy']
    print 'Done.'
    return out['accuracy']

if __name__ == '__main__':

    #file = open(filePath + "_trees_" + str(trees) +"_CNNRFdescription.txt",'w')
    result_file = open('/home/para/gitstore/HSICNN/vgg/tools/result.txt','w')
    for mark in range(51):
        if mark != 0:
            result = Test(mark)
            result_file.write('-----------------------------------------------------------\n')
            result_file.write('数据集文件夹 ' + str(mark) + ':      ' + str(result) + '\n')
    result_file.close()
