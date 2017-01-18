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
#mix_model_svm_ratio是为了以后采用组合混合模型的时候，保存一个svm在所有模型中的占比。以后根据需求进行扩充。。。TODO
mix_model_svm_ratio = 0
file_name, neighbors = data_util.prepare()

print "now constructing the network..."
#print "enter the layers each convolutional kernel covers: "
#neighbors = int(raw_input(prompt))
neighbors = neighbors + 1

print "the neighbors strategy is: " + str(neighbors)
print "enter the number of convolutional neurons:"
neurons = int(raw_input(prompt))
print "enter the number of neurons after the maxpooling layer:"
maxpoolings = int(raw_input(prompt))
print "enter the number of full layers\' neurons, default is 100:"
fullLayers = int(raw_input(prompt))
#if tempfullLayers > 1:
#    fullLayers = tempfullLayers
print "enter the batch size for bsgd:"
batch_size = int(raw_input(prompt))
print "enter the learning ratio:"
learning = float(raw_input(prompt))
print "enter the train decay:"
train_decay = float(raw_input(prompt))
print "enter the epoches you want the network to be trained:"
epoches = int(raw_input(prompt))

print "now choose the following strategy after the cnn network been trained:"
print "#1:train a cnn-svm joint framework;"
print "#2:train a cnn-rf joint framework;"
print "#3:train both cnn-svm and cnn-rf joint frameworks;"
print "#4:TODO: train a mix assemble cnn-classifier model."
following_strategy = int(raw_input(prompt))
if following_strategy == 4:
    print "enter the ratio of svm classifier:"
    mix_model_svm_ratio = int(row_input(prompt))

tress = 0
if following_strategy == 2 or following_strategy == 3:
    print "enter the count of trees you want to set in Random Forest:"
    trees = int(raw_input(prompt))

print "starting ..."
HSICNN.run_network(file_name, neurons, neighbors, maxpoolings, fullLayers,batch_size, learning, train_decay, epoches)

print "the training of the network have done."

if following_strategy == 1:
    #CNN + SVM
    print "now processing the cnn + svm joint framework..."
    cnnsvm.run(file_name, neurons, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
elif following_strategy == 2:
    #CNN + rfind
    print "now processing the cnn + rf joint framework..."
#    print "enter the count of trees you want to set in Random Forest:"
#    trees = int(raw_input(prompt))
    cnnrf.run(file_name,trees, neurons, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)
elif following_strategy == 3:
    #CNN+svm and CNN+RF
    
    print "now processing the cnn + svm joint framework..."
    cnnsvm.run(file_name, neurons, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)

    print "now processing the cnn + rf joint framework..."
    cnnrf.run(file_name, trees, neurons, neighbors, maxpoolings, fullLayers, batch_size, learning, train_decay)

file = open(file_name + "_experiment_description.txt", 'w')
file.write("-------------Experiment Description-------------\n")
file.write("Data set:" + file_name + "#\n")
file.write("neighbor strategy:" + str(neighbors) + "#\n")
file.write("Convolutional Neurons:" + str(neurons) + "#\n")
file.write("Max Polling Layer Neuron number:" + str(maxpoolings) + "#\n")
file.write("Full Layer Neuron number:" + str(fullLayers) + "#\n")
file.write("Batch size of SGD training:" + str(batch_size) + "#\n")
file.write("Training epoches of deep CNN:" + str(epoches) + "#\n")
file.write("Learning ratio:" + str(learning) + "#\n")
file.write("Train decay:" + str(train_decay) +"#\n")
if following_strategy == 2 or following_strategy == 3:
    file.write("Number of trees in random forest: " + str(trees) + "#\n")
file.write("===============================================")


