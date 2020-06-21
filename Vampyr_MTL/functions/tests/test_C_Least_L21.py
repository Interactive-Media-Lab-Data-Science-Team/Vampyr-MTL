from .test_data import get_data
from ..MTL_Cluster_Least_L21 import MTL_Cluster_Least_L21
from ...evaluations.utils import MTL_data_extract, MTL_data_split, opts
import numpy as np
import math

# iris data
X_train, X_test, Y_train, Y_test, df = get_data()
opts = opts(1500,2)
opts.tol = 10**(-6)


# customized data
clus_var = 900
task_var = 16
nois_var = 150

clus_num = 2
clus_task_num = 10
task_num = clus_num * clus_task_num
sample_size = 100
dimension = 20
comm_dim = 2
clus_dim = math.floor((dimension - comm_dim)/2)

# generate cluster model
cluster_weight = np.random.randn(dimension, clus_num)* clus_var
for i in range(clus_num):
    bll = np.random.permutation(range(dimension-clus_num))<=clus_dim
    blc = np.array([False]*clus_num)
    bll = np.hstack((bll, blc))
    cluster_weight[:,i][bll]=0
cluster_weight[-1-comm_dim:, :]=0
W = np.tile(cluster_weight, (1, clus_task_num))
cluster_index = np.tile(range(clus_num), (1, clus_task_num)).T

# generate task and intra-cluster variance
W_it = np.random.randn(dimension, task_num) * task_var
for i in range(task_num):
    bll = np.hstack(((W[:-1-comm_dim+1,i]==0).reshape(1,-1), np.zeros((1,comm_dim))==1))
    W_it[:,i][bll.flatten()]=0
W = W+W_it

W = W + np.random.randn(dimension, task_num)*nois_var

X = [0]*task_num
Y = [0]*task_num
for i in range(task_num):
    X[i] = np.random.randn(sample_size, dimension)
    xw = X[i] @ W[:,i]
    s= xw.shape
    xw = xw + np.random.randn(s[0]) * nois_var
    Y[i] = np.sign(xw)

class Test_CMTL_Least_classification(object):
    def test_basic_mat(self):
        clf = MTL_Cluster_Least_L21(opts, 3)
        clf.fit(X, Y)
        corr = clf.analyse()
        print(corr)
        
    
    def test_iris_accuracy(self):
        clf = MTL_Cluster_Least_L21(opts, 3)
        clf.fit(X_train, Y_train)
        corr = clf.analyse()
        print(corr)