from ..MTL_Softmax_L21_hinge import MTL_Softmax_L21
import numpy as np
import pandas as pd
from sklearn import datasets
from ...evaluations.utils import MTL_data_extract, MTL_data_split

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
y2 = np.array([0, 0, 10, 1])
y3 = np.array([1, 20, 1, 10])
Y = [y1, y2, y3]

# first ==1: a, first <=6: b, first >6: c, first <0: d, first >10: e
y1_c = np.array(['a', 'b', 'a', 'a', 'a', 'd', 'a'])
y2_c = np.array(['a', 'a', 'c', 'e'])
y3_c = np.array(['b', 'a', 'c', 'a'])
Y_c = [y1_c, y2_c, y3_c]

test = []
test.append(np.array([[2,1,1,1,1], [1,1,1,0,0], [250, 700, 100, 10, 20], [200, 230, 240, -1000, -2000], [1,2,3,4,5]]))
test.append(np.array([[2,2,5,11,200], [1000, 0, 200, 10, -10]]))
test.append(np.array([[4,3,4,3,74],[2,1,90,10,220], [2000, 200, 20, 10, 1], [-10, -1, -20, 1, 20], [1000, 2000, 3000, -1, -2], [202, 430, 12, -10, -1000]]))
target = np.array([[0, 0, 2, 18, 0], [0, 17], [1,0, 0, 0, 3, 0]])
target_c = np.array([['d', 'd', 'd', 'b', 'd'], ['a', 'd'], ['d', 'd', 'd', 'd', 'd', 'd']])

## iris data
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df['cat1']=0
df['cat2']=0
df['target'] = df['target'].astype(int)
df.loc[df['petal width (cm)']<=0.8, 'cat1'] = 0
df.loc[(df['petal width (cm)']>0.8) & (df['petal width (cm)']<=1.6), 'cat1'] = 1
df.loc[(df['petal width (cm)']>1.6) & (df['petal width (cm)']<=2.4), 'cat1'] = 2
df.loc[df['petal length (cm)']<=2.3, 'cat2'] = 0
df.loc[(df['petal length (cm)']>2.3) & (df['petal length (cm)']<=4.6), 'cat2'] = 1
df.loc[(df['petal length (cm)']>4.6) & (df['petal length (cm)']<=6.9), 'cat2'] = 2
X_i, Y_i = MTL_data_extract(df, 'cat2', 'target')
X_train, X_test, Y_train, Y_test = MTL_data_split(X_i, Y_i, test_size=0.4)

df2 = df.copy()
df2.loc[df['target']==0, 'target'] = 'flower1'
df2.loc[df['target']==1, 'target'] = 'flower2'
df2.loc[df['target']==2, 'target'] = 'flower3'
X_i, Y_i = MTL_data_extract(df2, 'cat2', 'target')
X_train_c, X_test_c, Y_train_c, Y_test_c = MTL_data_split(X_i, Y_i, test_size=0.4)


class Test_softmax_classification(object):
    def test_soft_numerical_accuracy(self):
        ult_thres = 0.5
        thres = 0.5
        its = 10
        succ = 0
        for it in range(its):
            clf = MTL_Softmax_L21(opts)
            clf.fit(X_train, Y_train, rho=0.00001)
            pred = clf.predict(X_test)
            correct = 0
            total = 0
            for i, j in zip(Y_test, pred):
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
    
    def test_soft_cat_accuracy(self):
        ult_thres = 0.5
        thres = 0.5
        its = 10
        succ = 0
        for it in range(its):
            clf = MTL_Softmax_L21(opts)
            clf.fit(X_train_c, Y_train_c, rho=0.00001)
            pred = clf.predict(X_test_c)
            correct = 0
            total = 0
            for i, j in zip(Y_test_c, pred):
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
