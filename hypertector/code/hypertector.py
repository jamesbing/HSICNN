#!/usr/bin/env python
# coding=utf-8
# O(∩_∩)O @ zhoujing @
# author @ jiabing leng @ nankai university @ tadakey@163.com
import HSICNN
import data_util
import cnnsvm
import cnnrf
from sys import argv

prompt = '>'
fullLayers = 100
file_name, neighbors = data_util.prepare()
print "now constructing the network..."
#print "enter the layers each convolutional kernel covers: "
#neighbors = int(raw_input(prompt))
neighbors = neighbors + 1
print "the neighbors strategy is: " + str(neighbors)
print "enter the number of neurons after the maxpooling layer:"
maxpoolings = int(raw_input(prompt))
print "enter the number of full layers\' neurons, default is 100:"
tempfullLayers = int(raw_input(prompt))
if tempfullLayers > 1:
    fullLayers = tempfullLayers
print "enter the batch size for bsgd:"
batch_size = int(raw_input(prompt))
print "enter the learning ratio:"
learning = float(raw_input(prompt))
print "enter the train decay:"
train_decay = float(raw_input(prompt))
print "enter the epoches you want the network to be trained:"
epoches = int(raw_input(prompt))
print "starting ..."

HSICNN.run_network(file_name, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)

#CNN + SVM
print "now processing the cnn + svm joint framework..."
cnnsvm.run(file_name, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)

#TODO: CNN + rf


