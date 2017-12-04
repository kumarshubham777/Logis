__author__ = 'Shubham'
import pandas as pd
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
max=0
fields=['DEFECT']
datacol=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fields)

datacol['DEFECT']=datacol['DEFECT'].replace(['yes','no'],[1,0])
y=datacol;
fieldsrow=['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3','LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC']
datarow=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fieldsrow)
x=datarow;

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
kf=cross_validation.KFold(250,n_folds=10,shuffle=False,random_state=None)
cnf=np.zeros(2)
accuracy=0

for train,test in kf:
 confusion = confusion_matrix(y_test,y_pred)
 cnf=cnf+confusion
 accuracy=accuracy+accuracy_score(y_test, y_pred)



print(cm)
print('Accuracy:' ,accuracy/10)









'''
from sklearn import cross_validation
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)



classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
kf=cross_validation.KFold(250,n_folds=10,shuffle=False,random_state=None)
cnf=np.zeros(2)
accuracy=0
for train,test in kf:

    confusion = confusion_matrix(y_test,y_pred)
    cnf=cnf+confusion
    accuracy=accuracy+accuracy_score(y_test, y_pred)
print(cnf)
print(accuracy/10)
'''