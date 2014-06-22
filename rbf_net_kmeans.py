''''Radial Basis Function Neural Network with K-Means averaging allotment of weights'''

import numpy as np
from scipy import *
from scipy.linalg 				 import norm, pinv
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from sklearn 					 import preprocessing
from sklearn.cluster 			 import KMeans
from sklearn.metrics    		 import f1_score

class RBFNN:

	def __init__(self, indim, outdim, numRBFCenters, RBFcenters, distance_std):
		self.indim = indim
		self.outdim = outdim
		self.numRBFCenters = numRBFCenters
		self.beta = distance_std # array comprising of standard deviations of points in each cluster
		self.RBFcenters = RBFcenters # the array of the centers of the |numRBFCenters| radial functions
		self.W = random.random((self.numRBFCenters, self.outdim))

	def _basisfunc(self, c, d, sigma):
		# Calculates the gaussian bell/radial basis function, the center of each function and sigma/width is deffernet.
		# Take care if sigma is zero!!
		assert len(d) == self.indim
		return exp(-norm(c-d)**2 / (2*sigma*sigma)  ) # the multiplication of 2 played a significant role

	def _calcAct(self, X):
	    # calculate activations of RBFs
		M = zeros((X.shape[0], self.numRBFCenters), float)
		for ci, c in enumerate(self.RBFcenters):
			for xi, x in enumerate(X):
				M[xi,ci] = self._basisfunc(c, x, self.beta[ci])
		return M
   
	def train(self, X, Y):
		""" X: matrix of dimensions n x indim
		y: column vector of dimension n x outdim """ 

		#print "center", self.RBFcenters
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
	
	numRBFCenters = 50 
	
	kmeans=KMeans(n_clusters=numRBFCenters) # KMeans to find the centroids for the RBF neurons.
	kmeans.fit(alltraindata['input'])
	centers=kmeans.cluster_centers_
	#centers.shape = (numRBFCenters,13)
	cluster_distance=kmeans.transform(alltraindata['input'])
	#cluster_distance.shape = (152,10) and kmeans.labels_.shape = (152,)

	# Calculating the sigma/smoothness parameter of each Radial Basis Function
	# It is the variance/standard deviation of the points of each cluster, thus giving a value for each RBFcenter
	distance_std=[] 
	distance_within_cluster=[]
	for lab in range(numRBFCenters):
		for x,label in enumerate(kmeans.labels_):
			if label == lab:
				distance_within_cluster.append(cluster_distance[x][label])	
		distance_std.append(np.std(distance_within_cluster))

	rbf=RBFNN(alltraindata.indim, alltraindata.outdim, numRBFCenters, centers, distance_std) # Passing the centers array for RBFNN initialization

	rbf.train(alltraindata['input'],alltraindata['target'])
	
	testdata_target=rbf.test(alltestdata['input']) #values obtained after testing, T is a 'n x outdim' matrix
	testdata_target = testdata_target.argmax(axis=1)  # the highest output activation gives the class. Selects the class predicted

	traindata_target=rbf.test(alltraindata['input'])	
	traindata_target = traindata_target.argmax(axis=1) # the highest output activation gives the class. Selects the class predicted
	
	#compare to y_test to obtain the accuracy.
	
	# count=0
	# for x in range(len(y_test)):
	# 	if testdata_target[x] == y_test[x]:
	# 		count+=1
	# tstresult2=float(count)/float(len(y_test)) * 100

	trnresult = percentError(traindata_target,alltraindata['class'])
   	tstresult = percentError(testdata_target,alltestdata['class'])
   	
	print "Accuracy on train data is: %5.2f%%," % (100-trnresult)
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

if  __name__ == "__main__":
	main()
