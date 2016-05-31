#!/usr/bin/env python
# coding=utf-8
##############################
#本文件提供公共工具方面的功能#
##############################
import scipy.io as sio
import numpy
import random
import theano


def randomNdarray(src1, src2):
    assert type(src1) == type(src2)
    assert src1.shape[0] == src2.shape[0]
    seq = range(src1.shape[0])
    random.shuffle(seq);
    des1 = numpy.ndarray(src1.shape)
    des2 = numpy.ndarray(src2.shape)
    for i in xrange(src1.shape[0]):
        des1[i] = src1[seq[i]]
        des2[i] = src2[seq[i]]
    return des1, des2

def normalizeNdarray(datas):
    minValue = 0
    maxValue = 10000
    return 2.0 * (datas - minValue) / (maxValue - minValue) - 1.0

def showXYData(x, y, imageName = ''):
    assert len(x) > 0 and len(y) > 0 and len(x) == len(y)
#    dpi = 72.
#    width = 300
#    height = 300
#    g = G.figure(figsize = (width/dpi, height/dpi))
#    G.clf()
#    G.cla()
#    G.axis([1, max(x), 0, 100])
#    G.xticks(xrange(0, int(max(x)+1), max(1, int(max(x) / 10.0))), size = 10)
#    G.yticks(xrange(0, 101, 10), size = 10)
    #G.title('accuracy_time(VegetationReducedData1500_10)')
#    G.title('Indian Pines dataset', size = 10)
#    G.xlabel('time(min)', size = 10)
#    G.ylabel('accuracy(%)', size = 10)
#    G.plot(x, y, 'k-')
#   axis = G.gca()
#   xaxis = axis.xaxis
#   xaxis.grid(False)
#   yaxis = axis.yaxis
#   yaxis.grid(True)
    #G.legend(loc = 5, fontsize = 10)
#    if imageName is not '':
#        G.savefig(imageName)
#    G.show()

def showData(datas, label):
    assert len(datas) > 0
#    minValue = 0
#    maxValue = 10000
#    dpi = 72.
#    width = 240
#    height = 160
#    try:
#        G.figure(figsize = (width/dpi, height/dpi))
#        G.clf()
#        G.cla()
#        G.axis([0, len(datas[0])-1, 0, maxValue])
#        G.xticks(xrange(0, len(datas[0])-1, 50), size = 10)
#        G.yticks(xrange(0, maxValue, 1000), size = 10)
#        G.title('Class' + str(label), size = 10)
#        G.xlabel('x')
#        G.ylabel('y')
#        for data in datas:
#            data = (data + 1.0) / 2.0 * (maxValue - minValue) + minValue
#            G.plot(data)
#        #G.legend(loc = 5, fontsize = 10)
#        G.savefig('./' + str(label) + '.png')
#        G.show()
#    except ImportError:
#        print 'Can not showTypeData'

def loadTestData(dataFile, typeId = -1):
    data = sio.loadmat(dataFile)
    
    test_data = data['DataTe']
    test_data = numpy.asarray(test_data, dtype = theano.config.floatX)

    test_label = data['CIdTe'][0,:]
    test_label = numpy.asarray(test_label, dtype = 'int32')

    return test_data, test_label


################################
#按照数据预处理的格式，装载数据#
################################
def loadData(dataFile, typeId = -1, bShowData = False):
    data = sio.loadmat(dataFile)

    train_data = data['DataTr']
    train_label = data['CIdTr'][0,:]
#    train_label = train_label[0,:]
#    return train_data,train_label
    train_set = [train_data, train_label]

    test_data = data['DataTe']
    test_label = data['CIdTe'][0,:]
    test_set = [test_data, test_label]

    valid_data = data['DataVa']
    valid_label = data['CIdVa'][0,:]
    valid_set = [valid_data, valid_label]

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y,dtype='int32'))
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def loadData_test(dataFile, typeId = -1, bShowData = False):
    data = sio.loadmat(dataFile)

    DataTrain = data['DataTrain']
    train_set_x = normalizeNdarray(DataTrain)
    CTrain = data['CTrain'][0,:]
    train_set_y = numpy.ndarray(0)
    for i in xrange(CTrain.shape[0]):
        temp = numpy.ndarray(CTrain[i])
        for j in xrange(CTrain[i]):
            if typeId == -1:
                temp[j] = i
            elif typeId == i:
                temp[j] = 0
            else:
                temp[j] = 1
        train_set_y = numpy.hstack((train_set_y, temp))
    return train_set_x, train_set_y

def loadData_former(dataFile, typeId = -1, bShowData = False):
    data = sio.loadmat(dataFile)

    DataTrain = data['DataTrain']
    train_set_x = normalizeNdarray(DataTrain)
    CTrain = data['CTrain'][0,:]
    train_set_y = numpy.ndarray(0)
    for i in xrange(CTrain.shape[0]):
        temp = numpy.ndarray(CTrain[i])
        for j in xrange(CTrain[i]):
            if typeId == -1:
                temp[j] = i
            elif typeId == i:
                temp[j] = 0
            else:
                temp[j] = 1
        train_set_y = numpy.hstack((train_set_y, temp))
    train_set_x, train_set_y = randomNdarray(train_set_x, train_set_y)

    if bShowData:
        for i in xrange(int(train_set_y.max()+1.5)):
            trainDatas = []
            for trainData, label in zip(train_set_x, train_set_y):
                if i == label:
                    trainDatas.append(trainData)
            showData(trainDatas, i+1)

    DataTest = data['DataTest']
    test_set_x = normalizeNdarray(DataTest);
    CTest = data['CTest'][0,:]
    test_set_y = numpy.ndarray(0)
    for i in xrange(CTest.shape[0]):
        temp = numpy.ndarray(CTest[i])
        for j in xrange(CTest[i]):
            if typeId == -1:
                temp[j] = i
            elif typeId == i:
                temp[j] = 0
            else:
                temp[j] = 1
        test_set_y = numpy.hstack((test_set_y, temp))
    test_set_x, test_set_y = randomNdarray(test_set_x, test_set_y)

    train_set = [train_set_x, train_set_y]
    test_set = [test_set_x, test_set_y]
    valid_set = [test_set_x, test_set_y]

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def saveStructure(fileName, convDims, convNodes, convKernels, convFilters, convPools, convLayers, fullNodes, fullLayers):
    fout = open(fileName, 'w')
    fout.write(str(convDims)+'\n')
    fout.write(str(convNodes)+'\n')
    fout.write(str(len(convLayers))+'\n')
    for i in xrange(len(convLayers)):
        fout.write(str(convKernels[i])+'\n')
        fout.write(str(convFilters[i])+'\n')
        fout.write(str(convPools[i])+'\n')
        fout.write(str(convLayers[i])+'\n')
    fout.write(str(len(fullLayers))+'\n')
    for i in xrange(len(fullLayers)):
        fout.write(str(fullNodes[i])+'\n')
        fout.write(str(fullLayers[i])+'\n')
    fout.close()

def loadStructure(fileName):
    convDims = 0
    convNodes = 0
    convKernels = []
    convFilters = []
    convPools = []
    convLayers = []
    fullNodes = []
    fullLayers = []
    fin = open(fileName, 'r')
    convDims = int(fin.readline())
    convNodes = int(fin.readline())
    for i in xrange(int(fin.readline())):
        convKernels.append(int(fin.readline()))
        convFilters.append(int(fin.readline()))
        convPools.append(int(fin.readline()))
        className = fin.readline()
        if className == str(layer.ConvPoolLayer) + '\n':
            convLayers.append(layer.ConvPoolLayer)
        if className == str(layer.FullLayer) + '\n':
            convLayers.append(layer.FullLayer)
        if className == str(layer.SoftmaxLayer) + '\n':
            convLayers.append(layer.SoftmaxLayer)
    for i in xrange(int(fin.readline())):
        fullNodes.append(int(fin.readline()))
        className = fin.readline()
        if className == str(layer.ConvPoolLayer) + '\n':
            fullLayers.append(layer.ConvPoolLayer)
        if className == str(layer.FullLayer) + '\n':
            fullLayers.append(layer.FullLayer)
        if className == str(layer.SoftmaxLayer) + '\n':
            fullLayers.append(layer.SoftmaxLayer)
    return convDims, convNodes, convKernels, convFilters, convPools, convLayers, fullNodes, fullLayers

def saveParams(fileName, params):
    data = {}
    for i in xrange(len(params)):
        if i % 2 == 0:
            data['W%d' %(i/2)] = params[i].get_value()
        else:
            data['b%d' %(i/2)] = params[i].get_value()
    sio.savemat(fileName, data)

def loadParams(fileName):
    data = sio.loadmat(fileName)
    params = []
    for i in xrange(len(data)-3):
        if i % 2 == 0:
            params.append(data['W%d' %(i/2)])
        else:
            params.append(data['b%d' %(i/2)])
    return params

def saveList(fileName, datas):
    file = open(fileName, 'w')
    file.write(str(len(datas)) + '\n')
    for data in datas:
        file.write(str(data) + '\n')
    file.close()

def loadList(fileName):
    datas = []
    file = open(fileName, 'r')
    for i in xrange(int(file.readline())):
        datas.append(float(file.readline()))
    file.close()
    return datas
