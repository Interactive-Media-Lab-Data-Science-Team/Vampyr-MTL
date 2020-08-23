from ..MTL_Softmax_L21 import MTL_Softmax_L21
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from ...evaluations.utils import MTL_data_extract, MTL_data_split, opts
from .test_data import get_data
from sklearn.linear_model import LogisticRegression
import os

opts = opts(1000,2)

X_train, X_test, Y_train, Y_test, df = get_data()

df2 = df.copy()
df2.loc[df['target']==0, 'target'] = 'flower1'
df2.loc[df['target']==1, 'target'] = 'flower2'
df2.loc[df['target']==2, 'target'] = 'flower3'
X_i, Y_i = MTL_data_extract(df2, 'cat2', 'target')
X_train_c, X_test_c, Y_train_c, Y_test_c = MTL_data_split(X_i, Y_i, test_size=0.4)

print(os.getcwd())
print('???????????????')
df3 = pd.read_csv('./cleaned_BRFSS.csv')

def normalize(X):
    for i in range(len(X)):
        min_max_scaler = preprocessing.MinMaxScaler()
        X[i] = min_max_scaler.fit_transform(X[i])
    return X

class Test_softmax_classification(object):
    def test_real_data(self):
        df4 = df3[(df3['ADDEPEV2']==2)|(df3['ADDEPEV2']==1)]
        # opts.tol = 1e-20
        X, Y = MTL_data_extract(df4, "ADDEPEV2", "_BMI5CAT")
        task = [0]*2
        taskT = 0
        for i in range(1):
            X_train, X_test, Y_train, Y_test = MTL_data_split(X, Y, test_size=0.998)
            X_train = normalize(X_train)
            X_test = normalize(X_test)
            for i in range(len(Y_train)):
                Y_train[i] = Y_train[i].astype(int)
            clf = MTL_Softmax_L21(opts)
            clf.fit(X_train, Y_train)
            pred = clf.predict(X_test)
            
            c_t = 0
            total = 0
            for i in range(len(pred)):
                correct = np.sum(pred[i]==Y_test[i])
                sub = len(pred[i])
                task[i] = max(task[i], correct/sub*100)
                total += sub
                c_t += correct
            taskT = max(taskT, c_t/total*100)
        print("accurcy for task 1 is {}%".format(task[0]))
        print("accurcy for task 2 is {}%".format(task[1]))
        print("total accuracy is {}%".format(taskT))
        
        for i in range(len(pred)):
            clf = LogisticRegression(random_state=0).fit(X_train[i], Y_train[i])
            s = clf.score(X_test[i], Y_test[i])
            print("SKLearn accuracy for task {} is {}%".format(i, s*100))
    
        assert c_t/total*100 == 0
    
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
            acc = clf.score(X_test_c, Y_test_c)
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
