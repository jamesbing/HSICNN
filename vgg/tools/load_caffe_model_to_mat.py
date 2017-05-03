#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.io as sio
import sys
import os

context = '/home/para/caffe/'
sys.path.insert(0,context + '/python')
import caffe
import csv

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

    print conv1_w
        
    os.mkdir('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/')
    
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv1_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv1_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv1_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv1_b)

    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv2_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv2_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv2_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv2_b)


    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv3_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv3_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv3_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv3_b)


    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv4_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv4_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv4_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv4_b)

    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv5_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv5_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv5_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv5_b)

    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv6_w.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv6_w)
    with open('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '/conv6_b.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(conv6_b)
    #net = caffe.Net('/home/para/gitstore/HSICNN/vgg/train_test.prototxt','/home/para/gitstore/HSICNN/vgg/experiment_results/' + str(experiment_number) + '/_iter_5000.caffemodel',caffe.TEST)
    sio.savemat('/home/para/gitstore/HSICNN/vgg/parameters/' + str(experiment_number) + '_parameters.mat', {'conv1_w':conv1_w,'conv1_b':conv1_b,'conv2_w':conv2_w,'conv2_b':conv2_b,'conv3_w':conv3_w,'conv3_b':conv3_b,'conv4_w':conv4_w,'conv4_b':conv4_b,'conv5_w':conv5_w,'conv5_b':conv5_b,'conv6_w':conv6_w,'conv6_b':conv6_b})
if __name__ == '__main__':

    for experiment_number in range(1,50):
        load(experiment_number)
    print "Save caffemodel to mat done."

