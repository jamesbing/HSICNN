#!/usr/bin/env python
# coding=utf-8
####################################
#本模块包含卷积神经网络中的必要组件#
####################################

from __future__ import print_function

__author__ = 'Jiabing Leng @ james ::: tadakey@163.c0m'

import os
import sys
import timeit
import math
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import tool

"""
构造卷积神经网络的函数
"""
def InitCNNModel(fileName, neighbor_strategy):

    batch_size = 9

    rng = numpy.random.RandomState(23455)
    datasets = tool.loadData(fileName)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #定义三个向量用于控制特征向量
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    #定义学习效率，该值决定的是寻找最优解时每一步下降的幅度，
    #也就是优化的步长，太大容易导致局部最优解困境，太小容易导致训练速度过慢
    learning_rate = 0.01


    #__todo__ = '现在先按照一个固定的结构来定义CNN，以后要改成更灵活、可配置的方式'
    
    #输入层结点，其大小与特征向量的维度一致
    input_nodes = train_set_x.get_value(borrow = True).shape[1]

    #构建第一个卷积层：在使用中，采用 input_nodes × 1的拉伸格式
 
    layer0_input = x.reshape((batch_size, 1, input_nodes, 1))
    layer0_conv_kernel_number = 20
    #该变量用于保存卷积跳步，其值等于采取的领域策略的数值+1
    kernel_jump_step = neighbor_strategy + 1
    layer0_conv_kernel_size = int(math.ceil(
        (input_nodes/kernel_jump_step/9.
        * kernel_jump_step)))

    n2_nodes_number = ((input_nodes - layer0_conv_kernel_size)/
                       kernel_jump_step + 1)

    n3_nodes_number = 40

    layer1_max_pool_kernel_size = math.ceil(
        float(n2_nodes_number / n3_nodes_number
    ))

   # max_pool_kernel_size = math.ceil()

    layer0 = ConvolutionalLayer(
        rng ,
        input = layer0_input,
        image_shape = (batch_size, 1, input_nodes, 1),
        filter_shape = (layer0_conv_kernel_number, 1,
                       layer0_conv_kernel_size, 1),
        poolsize = (int(layer1_max_pool_kernel_size),1)
        ) 

    #这一层是连接Convolutional Layer MaxPooling之后的那一层
    #与MaxPooling那一层和输出层连接的中间层次，其实是全连接层
    layer1_input = layer0.output.flatten(2)
    layer1 = HiddenLayer(
        rng,
        input = layer1_input,
        n_in = layer0_conv_kernel_number * n3_nodes_number * 1,
        n_out = 100,
        activation = T.tanh
        )

    #这一层是连接前面的全连接层以及后面的输出层之间的那一层连接
    #其操作实际上是一个logistics regression
    n_out_nodes = train_set_y.get_value(borrow = True).max() + 1
    layer2 = LogisticRegressionLayer(
        input=layer1.output, n_in = 100, n_out = n_out_nodes    
        )

    cost = layer2.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer2.errors(y),
        givens = {
            x:test_set_x[index * batch_size: (index + 1) * batch_size],
            y:test_set_y[index * batch_size: (index + 1) * batch_size]
        }
        )

    validate_model = theano.function(
        [index],
        layer2.errors(y),
        givens = {
            x:valid_set_x[index * batch_size: (index + 1) * batch_size],
            y:valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
        )
    
    params = layer2.params + layer1.params + layer0.params
    
    #为所有的权值参数创建一个梯度矩阵，以便用梯度下降算法进行训练
    grads = T.grad(cost, params)

    #权值更新法则
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    #根据更新法则以及相关参数，构造网络的基于反向传播梯度下降的训练函数
    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            x:train_set_x[index * batch_size: (index + 1) * batch_size],
            y:train_set_y[index * batch_size: (index + 1) * batch_size]
        }
        )
    ###############
    # TRAIN MODEL #
    ###############

    print('... training')
    
    n_train_batches = train_set_x.get_value(borrow = True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]
    n_test_batches = test_set_x.get_value(borrow = True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    #n_epochs这个值表示的是，对于网络的最大的优化次数
    n_epochs = 200
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    




"""
用于表征CNN中的每个层次的特征，用于构造神经网络
"""

class Layer(object):
    def __init__(self, layer_name, input_size, output_size, 
                nodes_per_map, number_of_maps, function_type = 'softmax'):
        self.layer_name = layer_name
        self.input_size = input_size
        self.output_size = output_size
        self.nodes_per_map = nodes_per_map
        self.maps = number_of_maps
        self.function_type = function_type



"""
定义CNN网络的基本架构
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

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for

class LogisticRegressionLayer(object):
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

        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
                )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    InitCNNModel('newPU1N.mat',1)
