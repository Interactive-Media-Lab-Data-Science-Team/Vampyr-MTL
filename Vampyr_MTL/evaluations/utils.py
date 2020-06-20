from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
