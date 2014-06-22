''''Radial Basis Function Neural Network with random allotment of weights'''

import numpy as np
from scipy import *
from scipy.linalg 				 import norm, pinv
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from sklearn 					 import preprocessing
from sklearn.metrics    		 import f1_score

class RBFNN:

	def __init__(self, indim, outdim, numRBFCenters):
		self.indim = indim
		self.outdim = outdim
		self.numRBFCenters = numRBFCenters
		self.centers = [random.uniform(-1, 1, indim) for i in xrange(numRBFCenters)] 
		# The centers have been chosen ramdomly, hence not so precise results. For better accuracy, 
		# the centers can be localized, using SMO or k-Means.
		self.beta = 8 # Is cross-validation/holdout method possible to estimate beta/smoothing parameter for better results.
		self.W = random.random((self.numRBFCenters, self.outdim))

	def _basisfunc(self, c, d):
		# Calculates the gaussian bell/radial basis function.
		assert len(d) == self.indim
		return exp(-self.beta * norm(c-d)**2)

	def _calcAct(self, X):
	    # calculate activations of RBFs
		M = zeros((X.shape[0], self.numRBFCenters), float)
		for ci, c in enumerate(self.centers):
			for xi, x in enumerate(X):
				M[xi,ci] = self._basisfunc(c, x)
		return M
   
	def train(self, X, Y):
		""" X: matrix of dimensions n x indim
		y: column vector of dimension n x outdim """ 
		# choose random center vectors from training set
		rnd_idx = random.permutation(X.shape[0])[:self.numRBFCenters]
		self.centers = [X[i,:] for i in rnd_idx]

		#print "center", self.centers
		# calculate activations of RBFs
		M = self._calcAct(X)
		#print M

		# calculate output weights (pseudoinverse)
		self.W = dot(pinv(M), Y)

	def test(self, X):
		""" X: matrix of dimensions n x indim """
		M = self._calcAct(X)
		Y = dot(M, self.W)
		return Y


def main():

	in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
	out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

	#getting in the data from csv files and making it suitable for further action.
	in_data=in_data[~np.isnan(in_data).any(1)]
	t=len(in_data[0,:])
	y_train=np.array(in_data[0:,t-1])
	x_train=np.array(in_data[0:,:t-1])

	scaler = preprocessing.StandardScaler().fit(x_train) #standardization plays an important role in all NN algos

	x_train=scaler.transform(x_train) #final x_train

	out_data=out_data[~np.isnan(out_data).any(1)]
	t=len(out_data[0,:])
	y_test=np.array(out_data[0:,t-1])
	x_test=np.array(out_data[0:,:t-1])

	x_test=scaler.transform(x_test) # final x_test

	alltraindata=ClassificationDataSet(t-1,1,nb_classes=2)
	for count in range(len((in_data))):
		alltraindata.addSample(x_train[count],[y_train[count]])

	alltraindata._convertToOneOfMany(bounds=[0,1])

	alltestdata=ClassificationDataSet(t-1,1,nb_classes=2)
	for count in range(len((out_data))):
		alltestdata.addSample(x_test[count],[y_test[count]])

	alltestdata._convertToOneOfMany(bounds=[0,1])
	
	numRBFCenters = 10 #the 'h' value
	
	rbf=RBFNN(alltraindata.indim, alltraindata.outdim, numRBFCenters)

	rbf.train(alltraindata['input'],alltraindata['target'])
	
	testdata_target=rbf.test(alltestdata['input']) #values obtained after testing, T is a 'n x outdim' matrix
	testdata_target = testdata_target.argmax(axis=1)  # the highest output activation gives the class. Selects the class predicted
  	#testdata_target = testdata_target.reshape(len(in_data),1)	

	#compare to y_test to obtain the accuracy.
	
	# count=0
	# for x in range(len(y_test)):
	# 	if testdata_target[x] == y_test[x]:
	# 		count+=1
	# tstresult2=float(count)/float(len(y_test)) * 100

   	tstresult = percentError(testdata_target,alltestdata['class'])
   	
	print "Accuracy on test data is: %5.2f%%," % (100-tstresult)

	for x in range(len(y_test)):
		if any(y_test[x]) == True:
			y_test[x] = 1
		else:
			y_test[x] = 0

	average_label = ['micro','macro','weighted']
	for label in average_label: 
		f1 = f1_score(y_test, testdata_target, average=label)
		print "f1 score (%s)" %label, "is ", f1

if __name__ == "__main__":
	main()
