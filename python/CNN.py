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
        #fan_out的值为：
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // 
                   numpy.prod(poolsize))
        
