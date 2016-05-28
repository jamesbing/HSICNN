#!/usr/bin/env python
# coding=utf-8
####################################
#本模块包含卷积神经网络中的必要组件#
####################################

__author__ = 'Jiabing Leng @ james ::: tadakey@163.c0m'

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import tool

"""
构造卷积神经网络的函数
"""
def InitCNNModel(fileName):
    rng = numpy.random.RandomState(23455)
    datasets = tool.loadData(fileName)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

"""
用于表征CNN中的每个层次的特征，用于构造神经网络
"""
class layer(object):



"""
定义CNN网络的基本构建
"""
class CNN(object):
    pass
#   def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2:2)):
#   __todo__ = '此处主要参考Yan Lecun的LeNet5进行构建，后续应该去LeNet并通用化'
"""
        根据调用时传导过来的各项参数构建CNN网络架构，init函数的参数应该修改
        """


"""
定义卷积层
"""
class ConvolutionalLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, 
                 poolsize = (2,2), needPool = True):
        """
        根据传入的参数初始化卷基层
        :tpye rng: numpy.random.RandomState
        :param rng: 用于初始化权值的随机种子

        :tpye input: theano.tensor.dtensor4
        :param input: 用于表征image_shape的形状

        :type filter_shape: 长度为4的tuple或者list
        :param filter_shape: (number_of_filters, number of input feature maps,
                                filter_height, filter_width)
        :type image_shape: 长度为4的tuple或者list
        :param image_shape:(batch_size, num of input feature maps,
                            image height, image width)

        :type poolsize: 长度为2的tuple或者list
        :param poolsize: （x,y）,x,y分别表示下采样的规模
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        #fan_out的值为：filter的个数 × filter的宽 × filter的高 ÷ 
        #    下采样核心的宽 × 下采样核心的高
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // 
                   numpy.prod(poolsize))
        
        #用随机种子初始化权值参数--Weights矩阵
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound, high = W_bound,
                         size = filter_shape),
            dtype = theano.config.floatX),
            borrow = True
            )

        #将全部bias赋值为0
        b_values = numpy.zeros((filter_shape[0],), 
                              dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)

        #调用theano的conv2d函数，对输入数据结构进行卷积构造
        conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            input_shape = image_shape
            )
       
#       if needPool == True:
           #进行下采样操作
        pooled_out = downsample.max_pool_2d(
               input = conv_out,
               ds = poolsize,
               ignore_border = True
               )

        self.output = T.tanh(pooled_out + self.b.dimshuffle(
            'x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

"""
定义Hidden Layer
这一层一般都是用于承前启后，用全连接的方式对高维数据进行映射
"""
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None,
                 activation = T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                    ),
                dtype = theano.config.floatX
                )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)

        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]


"""
定义logistic regression类
这一层用于将特征向量映射到分类平面上，输出一个分类向量
"""
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
