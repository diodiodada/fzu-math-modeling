import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

df = pd.read_csv('data.csv')

# ===============================

prices = df['lable']
features = df.drop('lable', axis = 1)

features = np.array(features)
prices = np.array(prices)

kf = KFold(653, n_folds=10)


for train_index, test_index in kf:
	# print("TRAIN:", train_index, "TEST:", test_index)

	X_train, X_test = features[train_index], features[test_index]
	y_train, y_test = prices[train_index], prices[test_index]

	# X_train = [ features[ii] for ii in train_index ] 
	# X_test  = [ features[ii] for ii in test_index ] 

	# y_train = [ prices[ii] for ii in train_index ] 
	# y_test  = [ prices[ii] for ii in test_index ] 

	clf = svm.SVC()
	clf.fit(X_train, y_train) 

	y_predict = clf.predict(X_test)

	acc = accuracy_score(y_predict, y_test)

	score = r2_score(y_predict, y_test)

	print("accuracy:")
	print( acc )

	print("r2_score:")
	print( score )

	print("")