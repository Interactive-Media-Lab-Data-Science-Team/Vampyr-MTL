from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import plotly.figure_factory as ff

class opts:
	def __init__(self, maxIter, init):
		self.maxIter = maxIter
		self.init = init
		self.pFlag = False

def MTL_data_split(X, Y, test_size=0.4, random_state=0):
	"""
		X: shape: t x n x d
		Y: shape: t x n x 1
	"""
	t = len(X)
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	for i in range(t):
		X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X[i], Y[i], test_size=test_size, random_state=random_state)
		X_train.append(X_train_s)
		X_test.append(X_test_s)
		y_train.append(y_train_s.flatten())
		y_test.append(y_test_s.flatten())
	return X_train, X_test, y_train, y_test

def MTL_data_extract(df, task_feat, target):
	'''
		df: extract from data frame
		task_feat: feature categorized as task
		target: feature that serves as target

		ret:
			X
			Y
	'''
	tasks = df[task_feat].unique()
	Y = []
	X = []
	for t in tasks:
		tmp1 = df.loc[df[task_feat]==t]
		tmp2 = tmp1.loc[:, tmp1.columns != task_feat]
		tmp = tmp2.loc[:, tmp2.columns != target]
		x = tmp.values
		X.append(np.array(x))
		y = tmp1.loc[:, df.columns == target].values
		Y.append(np.array(y))
	return X, Y

def RFA(df, task, target, top=10):
    def reformat(cols, w, top=10):
        RFA = OrderedDict()
        cols = np.array(cols)
        fet, task = w.shape
        total = {}
        all_tasks = []
        for i in range(task):
            col = w[:,i].flatten()
            index = sorted(range(len(col)), key=lambda i: col[i], reverse=True)[:top]
            e = set(cols[index])
            RFA["task {}".format(i+1)] = e
            all_tasks.append("task {}".format(i+1))
            total = set.union(e, total)
        print("all top {} colns are {}".format(top, total))
        ret = defaultdict(lambda: [])
        df_v = pd.DataFrame(False, index=list(total), columns=all_tasks)
        df_v2 = pd.DataFrame(None, index = list(total), columns=[str(p+1) for p in range(len(all_tasks))])
        df_RFA = []
        for t in all_tasks:
            df_RFA.append(list(RFA[t]))
        for i in total:
            count = 1
            for k, v in RFA.items():
                if i in v:
                    ret[i].append(k)
                    df_v[k][i]=True
                    df_v2[str(count)][i] = int(k[-2:])
                    if(len(k)==6):
                        df_v2[str(count)][i] = int(k[-1])
                    count+=1
        return df_v, all_tasks, list(total), df_v2, df_RFA, RFA
        
    def sort_df(df):
        fet, tsk = df.values.shape
        ret = pd.DataFrame(None, columns=list(df.columns))
        ind = list(df.index)
        seq = []
        for i in range(tsk):
            for j in range(fet):
                if(np.count_nonzero(~np.isnan(list(df_v2.iloc[j].values)))==i+1):
                    ret.loc[len(ret)] = df_v2.iloc[j].values
                    seq.append(ind[j])
        ret = ret.rename(index={i:j for i,j in zip(range(fet), seq)})
        return ret
    
    def get_z_text(z, mp):
        x, y = z.shape
        ret = np.empty([x, y],dtype="S10")
        for i in range(x):
            for j in range(y):
                ret[i][j]=mp[z[i][j]]
            return ret.astype(str).tolist()
        
    all_col = (df.loc[:, (df.columns != target)&(df.columns != tasks)].columns).tolist()
    df_v, all_tasks, total, df_v2, RFA, index = reformat(all_col, mtl_clf.W, top=top)
    mp = {i+1:"Task_{}".format(i) for i in range(len(X))}
    mp[None] = ''
    mp[np.nan] = ''
    df_v3 = sort_df(df_v2)
    z_text = get_z_text(df_v3.values, mp)
    fig = ff.create_annotated_heatmap(z = df_v3.values.tolist(), annotation_text=z_text, y=list(df_v3.index))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    return fig
	
