from ..MTL_Least_L21 import MTL_Least_L21
import numpy as np
from ..utils import MTL_data_extract, MTL_data_split, opts

opts = opts(100,2)
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

t1 = np.array([[1,2,3,4,5],[6,7,8,9,0]])
t2 = np.array([[1,2,3,4,6]])
t3 = np.array([[4,3,4,3,4],[1,2,1,4,5],[7,8,9,0,1]])
X2 = [t1, t2, t3]
Y2 = [np.array([15, 30]), np.array([16]), np.array([18,13,25])]

class Test_linear_summation(object):
    
    def test_FGLasso_projection(self):
        clf = MTL_Least_L21(opts)
        D = np.array([[1,2,3],[4,5,6]])
        lbd = 0.01
        ret = clf.FGLasso_projection(D, lbd)
        np.testing.assert_array_almost_equal(ret, np.array([[0.9973, 1.9947, 2.9920],[3.9954, 4.9943, 5.9932]]), decimal=3)
        
    def test_non_smooth(self):
        clf = MTL_Least_L21(opts)
        W = np.array([[1,2,3, 4, 5],[6,7,8, 9, 0]])
        lbd = 0.01
        ret = clf.nonsmooth_eval(W, lbd)
        np.testing.assert_almost_equal(ret, 0.2258, decimal=3)
        
    def test_linear_regression(self):
        clf = MTL_Least_L21(opts)
        clf.fit(X, Y, rho=0.0001)
        pred = clf.predict(test)
        print(pred)
        np.testing.assert_allclose(pred, target, rtol=0.1)
    
    def test_linear_regression2(self):
        t = np.array([[0.5960,0.2421,0.9588],[0.8378,0.4843,0.9698],[1.0798,0.7268,1.0442],[1.3217,0.9692,0.8458],[0.8540,1.4539,1.1345]])
        clf = MTL_Least_L21(opts)
        clf.fit(X2, Y2, rho=0.001)
        np.testing.assert_array_almost_equal(clf.W, t, decimal=2)
    
