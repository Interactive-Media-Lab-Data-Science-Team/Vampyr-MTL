from ..MTL_Least_L21 import MTL_Least_L21
import numpy as np
import pytest

class opts:
	def __init__(self, maxIter, init):
		self.maxIter = maxIter
		self.init = init
		self.pFlag = False

opts = opts(100,0)
task1 = np.array([[1,2,3,4,5],[6,7,8,9,0], [1,2,3,4,6], [1,2,3,4,6]])
task2 = np.array([[1,2,3,4,6], [1,2,3,4,6]])
task3 = np.array([[4,3,4,3,4], [1,2,1,4,5], [7,8,9,0,1], [1,2,3,4,6]])
X = [task1, task2, task3]

y1 = np.array([15, 30, 16, 17])
y2 = np.array([18, 16])
y3 = np.array([18, 13, 26, 16])
Y = [y1, y2, y3]

test = []
test.append(np.array([2,1,1,1,1]))
test.append(np.array([2,2,5,11,2]))
test.append(np.array([2,1,90,10,20]))
target = np.array([[6, 22, 123]]).reshape((-1,1))

class Test_linear_summation(object):
    def test_linear_regression(self):
        clf = MTL_Least_L21(opts)
        clf.fit(X, Y, rho=0.0001)
        pred = clf.predict(test)
        np.testing.assert_allclose(pred, target, rtol=0.1)
