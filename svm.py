import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

train_size = 400

df = pd.read_csv('data.csv')

prices = df['lable']
features = df.drop('lable', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.2,random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train) 

y_predict = clf.predict(X_test)

acc = accuracy_score(y_predict, y_test)

score = r2_score(y_test, y_predict)

print("accuracy:")
print( acc )

print("r2_score:")
print( score )





