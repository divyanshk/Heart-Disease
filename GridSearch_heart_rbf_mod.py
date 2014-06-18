#GridSearch to find the best parameters for the model

import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report 
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing


in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

def GridSearch_rbf(in_data,out_data):
    
    print("Running GridSearch_rbf!!")

    in_data=in_data[~np.isnan(in_data).any(1)]
    print in_data.shape
    t=len(in_data[0,:])
    Y=np.array(in_data[0:,t-1])
    X=np.array(in_data[0:,:t-1])
    print np.isnan(X).any()
    
    #Find the best parameters for the model using cross-validation
    tuned_parameters=[{'kernel' : ['rbf'], 'gamma' : [1e-8, 1e-7,1e-6, 1e-5],
                       'C' : [10, 1000, 1e5, 1e7]}]

    # scaler = preprocessing.StandardScaler().fit(X) #standardization
    # X_scaled=scaler.transform(X) 
    #normalizer = preprocessing.Normalizer().fit(X)  #normalization
    #X_scaled=normalizer.transform(X)

    clf=GridSearchCV(SVC(), tuned_parameters, cv=10, scoring='f1')
    clf.fit(X,Y)
    print("Best parameters set found on development set:")
    print"**********"
    print(clf.best_estimator_)
    print"**********"
    print("Grid scores on development set:")
    print"**********"
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print"**********"

    print("Detailed classification report:")
    print"**********"
    print("The model is trained on the full development set.")

    #Accuracy in training data (1-Ein)
    print "Training Data Accuracy =" +str(clf.score(X,Y))

    #Verify model using Test data
    out_data=out_data[~np.isnan(out_data).any(1)]
    print out_data.shape
    t=len(out_data[0,:])
    Yt=np.array(out_data[0:,t-1])
    Xt=np.array(out_data[0:,:t-1])
    print np.isnan(Xt).any()

    #Xt_scaled=scaler.transform(Xt)
    #Xt_scaled=normalizer.transform(Xt)
    
    print("The test scores are computed on the full evaluation(test) set.")
    print"**********"
    
    #Accuracy in test data (1-Eout)
    print "Test Data Accuracy =" +str(clf.score(Xt,Yt))

    #print ("Number of support vectors = " + str(clf.n_support_))

    #print ("The support vectors are " + str(clf.support_vectors_))

    target_names=['class 0', 'class 1']
    print(classification_report(Yt,clf.predict(Xt),target_names=target_names))
    print"**********"
    
if __name__ == "__main__":
   GridSearch_rbf(in_data,out_data)    

    
