__author__ = 'Shubham'
from sklearn import cross_validation
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
fields=['DEFECT']
datacol=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fields)

datacol['DEFECT']=datacol['DEFECT'].replace(['yes','no'],[1,0])
y=datacol;
def func(a):
    if a==0:
        return 'WMC'
    if a==1:
        return 'DIT'
    if a==2:
        return 'NOC'
    if a==3:
        return 'CBO'
    if a==4:
        return 'RFC'
    if a==5:
        return 'LCOM'
    if a==6:
        return 'Ca'
    if a==7:
        return 'Ce'
    if a==8:
        return 'NPM'
    if a==9:
        return 'LCOM3'
    if a==10:
        return 'LOC'
    if a==11:
        return 'DAM'
    if a==12:
        return 'MOA'
    if a==13:
        return 'MFA'
    if a==14:
        return 'CAM'
    if a==15:
        return 'IC'
    if a==16:
        return 'CBM'
    if a==17:
        return 'AMC'

def LR(list):
 temp=[]
 cntr=0
 for i in list:
     if i==1:
         val=func(cntr)
         temp.append(val)

     cntr += 1
 if temp==[]:
        return 0.0
    #print(temp)
 x=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=temp)
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
 sc_x=StandardScaler()
 x_train=sc_x.fit_transform(x_train)
 x_test=sc_x.fit_transform(x_test)



 classifier=LogisticRegression(random_state=0)
 classifier.fit(x_train,y_train)

 y_pred=classifier.predict(x_test)

 from sklearn.metrics import confusion_matrix
 cm=confusion_matrix(y_test,y_pred)



 kf=cross_validation.KFold(250,n_folds=10,shuffle=False,random_state=None)
 cnf=np.zeros(2)
 accuracy=0
 for train,test in kf:

    confusion = confusion_matrix(y_test,y_pred)
    cnf=cnf+confusion
    accuracy=accuracy+accuracy_score(y_test, y_pred)

 return accuracy/10

