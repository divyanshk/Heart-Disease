import numpy as np
import random
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score



from copy import deepcopy


def main():
    tests=1
    Ain,Aout=0,0
    f1score=0
    Coef=[[0,0,0,0,0,0,0,0,0,0,0,0,0]]
    new_data=np.vstack([np.genfromtxt('logit-train.csv', delimiter = ','), np.genfromtxt('logit-test.csv', delimiter = ',')])
    
    #print new_data.shape
    new_data=new_data[~np.isnan(new_data).any(1)]
    t=len(new_data[0,:])

    sss=StratifiedShuffleSplit(new_data[:,t-1], tests, test_size=0.2, random_state=0)
    for train_index, test_index in sss:
        X,Xt = new_data[train_index, :t-1], new_data[test_index, :t-1]
        Y,Yt = new_data[train_index, t-1], new_data[test_index, t-1]
        #Build Logistic Regression Model
        param_grid=[{'kernel' : ['rbf'], 'gamma' : [1e-8, 1e-7,1e-6, 1e-5],
                           'C' : [10, 1000, 1e5, 1e7]},
                          {'kernel' : ['linear'], 'C' : [1e-6,1e-4,1e-3,1e-2,1e5,1e7]}]

        lr=GridSearchCV(SVC(), param_grid, scoring='f1', cv=5)
        lr.fit(X,Y)
        #print (lr.best_estimator_)
        #print()

        #In-sample Accuracy
        #P=lr.predict_proba(X)
        Z=lr.predict(X)
        Ain+=lr.score(X,Y)
        
        #Out-of-sample Accuracy
        #Pt=lr.predict_proba(Xt)
        Zt=lr.predict(Xt)
        Aout+=lr.score(Xt,Yt)
        #Coef+=lr.coef_
        target_names=['class 0', 'class 1']
        #print(classification_report(Yt,lr.predict(Xt),target_names=target_names))
        f1score+=f1_score(Yt,Zt)
        


    print " The Average Ain is " + str(float(Ain)/tests)
    print " The Average Aout is " + str(float(Aout)/tests)
    print " The Average F1 Score is " + str(float(f1score)/tests)

    
   # print " The Average coefs are " + str(float(Coef)/tests)
    
main()
