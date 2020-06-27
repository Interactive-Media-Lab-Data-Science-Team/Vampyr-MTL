from .test_data import get_data
from ..MTL_Cluster_Least_L21 import MTL_Cluster_Least_L21
from ...evaluations.utils import MTL_data_extract, MTL_data_split, opts
import numpy as np
import math
from scipy import linalg
import plotly.express as px

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
    """Pytest for Cluster Least Classification L21

    Args:
        object ([type]): entry point for pytest
    """
    
    def test_bsa_ihb(self):
        """ Test for bsa_ihb function inside CMTL usage
        """
        A = np.array([[1,2],[3,4]])
        EValue, EVector = linalg.eig(A) 
        Pz = np.real(EVector)
        # diag_EValue = np.real(np.diagonal(Evalue))
        diag_EValue = np.real(EValue).reshape((-1,1))
        clf = MTL_Cluster_Least_L21(opts, 3)
        x_star, t_star, it = clf.bsa_ihb(diag_EValue, 3)
        np.testing.assert_array_equal(x_star,
                              np.array([[0],[0]]))
        assert np.isnan(t_star) == True
        assert it == 3
    def test_basic_mat(self):
        """Test with basic matrix cases
        """
        clus_num = 2
        clus_task_num = 10
        task_num = clus_num * clus_task_num
        clf = MTL_Cluster_Least_L21(opts, 2)
        clf.fit(X, Y)
        corr = clf.analyse()
        # print(corr)
        # fig = px.imshow(corr, color_continuous_scale='Bluered_r')
        # fig.update_layout(
        # title={
        #     'text': "predict",
        #     })
        # fig.show()
        OrderedTrueModel = np.zeros(W.shape)
        clus_task_num = task_num//clus_num
        for i in range(clus_num):
            clusModel = W[:, i:task_num:clus_num]
            OrderedTrueModel[:, (i)*clus_task_num: (i+1)* clus_task_num] = clusModel
        corr2 = 1-np.corrcoef(OrderedTrueModel)
        # fig2 = px.imshow(corr2, color_continuous_scale='Bluered_r')
        # fig2.update_layout(
        # title={
        #     'text': "real",
        #     })
        # fig2.show()

    def test_check_simplified(self):
        """Test with simplified version matrix
        """
        # generate cluster model
        cluster_weight = np.ones((dimension, clus_num))* clus_var
        W = np.tile(cluster_weight, (1, clus_task_num))
        cluster_index = np.tile(range(clus_num), (1, clus_task_num)).T

        # generate task and intra-cluster variance
        W_it = np.zeros((dimension, task_num)) * task_var
        for i in range(task_num):
            bll = np.hstack(((W[:-1-comm_dim+1,i]==0).reshape(1,-1), np.zeros((1,comm_dim))==1))
            W_it[:,i][bll.flatten()]=0
        W = W+W_it

        W = W + np.zeros((dimension, task_num))*nois_var

        X = [0]*task_num
        Y = [0]*task_num
        for i in range(task_num):
            X[i] = np.ones((sample_size, dimension))
            xw = X[i] @ W[:,i]
            s= xw.shape
            xw = xw + np.ones((s[0])) * nois_var
            Y[i] = np.sign(xw)

        clf = MTL_Cluster_Least_L21(opts, 2)
        clf.fit(X, Y)
        corr = clf.analyse()
    # def test_iris_accuracy(self):
    #     clf = MTL_Cluster_Least_L21(opts, 3)
    #     clf.fit(X_train, Y_train)
    #     corr = clf.analyse()
    #     print(corr)