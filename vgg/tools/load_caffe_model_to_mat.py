#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.io as sio
import sys

context = '/home/para/caffe/'
sys.path.insert(0,context + '/python')
import caffe

def load(experiment_number):
    caffe.set_mode_gpu()
#    experiment_number = 1
    net = caffe.Net('/home/para/gitstore/HSICNN/vgg/train_test.prototxt','/home/para/gitstore/HSICNN/vgg/experiment_results/' + str(experiment_number) + '/_iter_5000.caffemodel',caffe.TEST)
    conv1_w = net.params['conv1'][0].data
    conv1_b = net.params['conv1'][1].data
    conv2_w = net.params['conv2'][0].data
    conv2_b = net.params['conv2'][1].data
    conv3_w = net.params['conv3'][0].data
    conv3_b = net.params['conv3'][1].data
    conv4_w = net.params['conv4'][0].data
    conv4_b = net.params['conv4'][1].data
    conv5_w = net.params['conv5'][0].data
    conv5_b = net.params['conv5'][1].data
    conv6_w = net.params['conv6'][0].data
    conv6_b = net.params['conv6'][1].data

    #net = caffe.Net('/home/para/gitstore/HSICNN/vgg/train_test.prototxt','/home/para/gitstore/HSICNN/vgg/experiment_results/' + str(experiment_number) + '/_iter_5000.caffemodel',caffe.TEST)
    sio.savemat('/home/para/gitstore/HSICNN/vgg/experiment_results/' + str(experiment_number) + '/parameters.mat', {'conv1_w':conv1_w,'conv1_b':conv1_b,'conv2_w':conv2_w,'conv2_b':conv2_b,'conv3_w':conv3_w,'conv3_b':conv3_b,'conv4_w':conv4_w,'conv4_b':conv4_b,'conv5_w':conv5_w,'conv5_b':conv5_b,'conv6_w':conv6_w,'conv6_b':conv6_b})
if __name__ == '__main__':
    experiment_number = 1
    load(experiment_number)
    print "Save caffemodel to mat done."

