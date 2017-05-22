import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

train_size = 400

df = pd.read_csv('data.csv')

x = np.array(df[['a1','a2','a3','a4','a5','a6','a7','a8','a9']][0:train_size]) 

y = np.array(df['lable'][0:train_size]) 

clf = svm.SVC()
clf.fit(x, y)  

x_test = np.array(df[['a1','a2','a3','a4','a5','a6','a7','a8','a9']][train_size:653]) 
y_test = np.array(df['lable'][train_size:653])

y_predict = clf.predict(x_test)

acc = accuracy_score(y_predict, y_test)

score = r2_score(y_test, y_predict)

print("accuracy:")
print( acc )

print("r2_score:")
print( score )







