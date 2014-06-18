import numpy as np
import random
from sklearn import svm
from sklearn.metrics import classification_report


def main():
    in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
    out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

    in_data=in_data[~np.isnan(in_data).any(1)]
    print in_data.shape
    t=len(in_data[0,:])
    Y=np.array(in_data[0:,t-1])
    X=np.array(in_data[0:,:t-1])
    print np.isnan(X).any()

    clf=svm.SVC(C=0.0001,kernel='poly',degree=2,gamma=1,coef0=1)
    clf.fit(X,Y)

    #Accuracy in training data (1-Ein)
    print "Training Data Accuracy =" +str(clf.score(X,Y))

    #Verify model using Test data
    out_data=out_data[~np.isnan(out_data).any(1)]
    print out_data.shape
    t=len(out_data[0,:])
    Yt=np.array(out_data[0:,t-1])
    Xt=np.array(out_data[0:,:t-1])
    print np.isnan(Xt).any()

    #Accuracy in test data (1-Eout)
    print "Test Data Accuracy =" +str(clf.score(Xt,Yt))

    print ("Number of support vectors = " + str(clf.n_support_))

    #print ("The support vectors are " + str(clf.support_vectors_))

    target_names=['class 0', 'class 1']
    print(classification_report(Yt,clf.predict(Xt),target_names=target_names))

main()
    

    
