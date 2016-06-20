#!/usr/bin/env python
# coding=utf-8

from tool import loadData
import numpy as np
from sklearn import cross_validation,decomposition,svm

train,valid,test = loadData("newIndian60percentN.mat")
X_train = train[0]
Y_train = train[1]
 
X_valid = valid[0]
Y_valid = valid[1]

X_test = test[0]
Y_test = test[1]

pca = decomposition.RandomizedPCA(n_components = 100,whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


clf = svm.SVC(C=1.2, kernel = 'linear',  gamma=0.0008, probability=True,
             tol = 0.000000001, verbose=True, max_iter = -1)
clf.fit(X_train, Y_train)
print "在测试集上的平均正确率为:"
print clf.score(X_test, Y_test)

#result = clf.predict(X_train)
#correctRatio = np.mean(np.equal(result,Y_train))
#print correctRatio


