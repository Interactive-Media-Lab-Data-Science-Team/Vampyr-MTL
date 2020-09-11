import numpy as np
import pandas as pd
from sklearn import datasets
from ..utils import MTL_data_extract, MTL_data_split, opts, MTL_ClusterBoosted_data_extract_auto

class Test_Data(object):
    def get_data(self):
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
        # X_i, Y_i = MTL_data_extract(df, 'cat2', 'target')
        X_i, Y_i = MTL_ClusterBoosted_data_extract_auto(df, 'cat2')
        X_train, X_test, Y_train, Y_test = MTL_data_split(X_i, Y_i, test_size=0.4)
        return X_train, X_test, Y_train, Y_test, df