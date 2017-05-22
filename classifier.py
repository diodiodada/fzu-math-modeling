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

df_lables = df['lable']
df_features = df.drop('lable', axis = 1)

features = np.array(df_features)
lables = np.array(df_lables)

kf = KFold(653, n_folds=10)

w1 = 0.3
w2 = 0.4
w3 = 0.3

acc = 0
score = 0

for train_index, test_index in kf:

	X_train, X_test = features[train_index], features[test_index]
	y_train, y_test = lables[train_index], lables[test_index] 

	# naive_bayes---1
	clf_1 = GaussianNB()
	clf_1.fit(X_train, y_train)
	y_predict_1 = clf_1.predict(X_test)


	# svm---2
	clf_2 = svm.SVC(kernel="linear")
	clf_2.fit(X_train, y_train) 
	y_predict_2 = clf_2.predict(X_test)

	# DecisionTreeClassifier---3
	clf_3 = tree.DecisionTreeClassifier()
	clf_3.fit(X_train, y_train)
	y_predict_3 = clf_3.predict(X_test)

	# add all
	y_predict_all = y_predict_1*w1+y_predict_2*w2+y_predict_3*w3
	y_predict = np.array(y_predict_all)

	for index in range(len(y_test)):
		if y_predict_all[index] > 0.5:
			y_predict[index]=1
		else:
			y_predict[index]=0


	acc += accuracy_score(y_predict, y_test)

	score += r2_score(y_predict, y_test)

print("accuracy:")
print( acc/10.0 )

print("r2_score:")
print( score/10.0 )