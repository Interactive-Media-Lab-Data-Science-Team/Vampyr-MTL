from ..MTL_Logistic_L21 import MTL_Logistic_L21
import numpy as np

class opts:
	def __init__(self, maxIter, init):
		self.maxIter = maxIter
		self.init = init
		self.pFlag = False

opts = opts(1000,0)

task1 = np.array([[1,2,3,4,5],[6,7,8,9,0], [1,2,3,4,6], [1,2,3,4,6], [1,0,1,2,1], [-1,-100,10,20,10], [1,1,-100,100, 1]])
task2 = np.array([[1,2,3,4,6], [1,2,3,4,6], [10,20,30,40,50], [1000, 0,100, 2,20]])
task3 = np.array([[4,3,4,3,74], [1,2,1,4,-100], [7,8,90,0,1], [1,2,3,4,-6]])
X = [task1, task2, task3]

# sum to 16: 0
y1 = np.array([0, 1, 0, 0, 0, 0, 0])
y2 = np.array([0, 0, 1, 1])
y3 = np.array([1, 0, 1, 0])
Y = [y1, y2, y3]

test = []
test.append(np.array([[2,1,1,1,1], [1,1,1,0,0], [250, 700, 100, 10, 20], [200, 230, 240, -1000, -2000]]))
test.append(np.array([[2,2,5,11,200], [1000, 0, 200, 10, -10]]))
test.append(np.array([[2,1,90,10,220], [2000, 200, 20, 10, 1], [-10, -1, -20, 1, 20], [1000, 2000, 3000, -1, -2], [202, 430, 12, -10, -1000]]))
target = np.array([[0, 0, 1, 0], [1, 1], [1, 1, 0, 1, 0]])


class Test_linear_classification(object):
    def test_binary_regression_accuracy(self):
        """Test with self identified matrix, structured for unstable cases
        """
        ult_thres = 0.7
        thres = 0.5
        its = 10
        succ = 0
        for it in range(its):
            clf = MTL_Logistic_L21(opts)
            clf.fit(X, Y, rho=0.00001)
            pred = clf.predict(test)
            correct = 0
            total = 0
            for i, j in zip(target, pred):
                for k,l in zip(i,j):
                    if(k == l):
                        correct+=1
                    total+=1
            acc = correct/total
            if(acc>thres):
                succ+=1    
                print("pass with accuracy {}".format(acc))
            else:
                print("fail with accuracy {}".format(acc))
        assert succ/its >=ult_thres
