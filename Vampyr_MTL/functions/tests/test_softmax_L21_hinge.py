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

opts = opts(1000,2)

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
        thres = 0.9
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
                pass
        print(">>>>>>>>>>>>>>>>>>>>>")
        print("in total of {} tests, {} passes {} threshold".format(its, acc, thres))
        print(">>>>>>>>>>>>>>>>>>>>>")
        assert succ/its >=ult_thres
    
    def test_soft_cat_accuracy(self):
        ult_thres = 0.5
        thres = 0.9
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
                # print("pass with accuracy {}".format(acc))
            else:
                # print("fail with accuracy {}".format(acc))
                pass
        print(">>>>>>>>>>>>>>>>>>>>>")
        print("in total of {} tests, {} passes {} threshold".format(its, succ/its, thres))
        print(">>>>>>>>>>>>>>>>>>>>>")
        assert succ/its >=ult_thres
