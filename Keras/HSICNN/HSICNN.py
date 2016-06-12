#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from keras.utils import np_utils



