from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one_loss
import numpy as np

from GridSearch_heart_rbf import GridSearch_rbf
 
'''hi all!!'''

def main(): 
	in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
	out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

	in_data=in_data[~np.isnan(in_data).any(1)]
	print in_data.shape
	t=len(in_data[0,:])
	y=np.array(in_data[0:,t-1])
	X=np.array(in_data[0:,:t-1])
	print np.isnan(X).any()

	'''Create the RFE object and compute a cross-validated score'''
	svc = SVC(kernel="linear")
	rfecv = RFECV(estimator=svc, step=2, cv=StratifiedKFold(y, 2),
	              scoring='accuracy')
	rfecv.fit(X, y)

	print("Optimal number of features : %d" % rfecv.n_features_)
	''' Tells which features to select, the ones that are the optima features are masked as 1.'''
	print rfecv.ranking_
	print rfecv.grid_scores_

	''' Plot number of features VS. cross-validation scores '''
	import pylab as pl
	pl.figure()
	pl.xlabel("Number of features selected")
	pl.ylabel("Cross validation score (nb of misclassifications)")
	pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	pl.show()

	'''Adding another one as the 14th column.'''
	rfecv.ranking_=np.append(rfecv.ranking_,1) 

	#Revising the X to include only the features considered optimal by rfecv
	in_data_revised=in_data[:,(rfecv.ranking_== 1)]
	print in_data_revised.shape
	
	'''Similarly, revise the out_data also'''
	out_data_revised=out_data[:,(rfecv.ranking_==1)]
	print out_data_revised.shape
	
	'''Call the GridSearch_rbf from GridSearch_heart_rbf for running the classification procedure on 
	the new set of features.'''
	
	GridSearch_rbf(in_data_revised,out_data_revised)

if __name__ == "__main__":
	main()