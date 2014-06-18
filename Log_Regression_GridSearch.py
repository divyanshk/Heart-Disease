import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from copy import deepcopy


def main():
    in_data = np.genfromtxt('logit-train.csv', delimiter = ',')
    out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

    #print in_data.shape
    in_data=in_data[~np.isnan(in_data).any(1)]
    #print in_data.shape
    t=len(in_data[0,:])
    #print in_data[0,0]== nan
    Y=np.array(in_data[0:,t-1])
    X=np.array(in_data[0:,:t-1])
    #print np.isnan(X).any()
    
    param_grid={'C' : [1e-5, 1e-8, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e5, 1e8]}
    
    lr=GridSearchCV(LogisticRegression(penalty='l2'), param_grid, scoring='f1', cv=10)
    lr.fit(X,Y)
    print (lr.best_estimator_)
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in lr.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()
    P=lr.predict_proba(X)

    Z=lr.predict(X)

    print "In-sample accuracy =" + str(lr.score(X,Y))

    #print out_data.shape
    out_data=out_data[~np.isnan(out_data).any(1)]
    #print out_data.shape
    l=len(out_data[0,:])
    #print in_data[0,0]== nan
    Yt=np.array(out_data[0:,l-1])
    Xt=np.array(out_data[0:,:l-1])
    #print np.isnan(Xt).any()

    Pt=lr.predict_proba(Xt)
    Zt=lr.predict(Xt)
    print sum(Zt!=Yt)

    print "Test data accuracy =" + str(lr.score(Xt,Yt))
    target_names=['class 0', 'class 1']
    print(classification_report(Yt,lr.predict(Xt),target_names=target_names))

    #print lr.coef_
    

main()
