from sklearn.model_selection import train_test_split

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
		X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, Y, test_size=test_size, random_state=random_state)
		X_train.append(X_train_s)
		X_test.append(X_test_s)
		y_train.append(y_train_s)
		y_test.append(y_test_s)
	return X_train, X_test. y_train, y_test