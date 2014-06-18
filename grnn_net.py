'''Generalised Regression Neural Network'''

import numpy as np
from scipy import *
from scipy.linalg 				 import pinv
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from sklearn 					 import preprocessing

class GRNN:

	def __init__(self, indim, outdim):
		self.indim = indim
		self.outdim = outdim

	def _distancefun(self, x, x_sample):
		assert len(x_sample) == self.indim
		return dot((x - x_sample) , transpose(x - x_sample))

	def _radialfun(self, x, x_sample, sigma):
		return exp( -self._distancefun(x, x_sample) / (2*sigma*sigma) )

	def predict(self, x, X, Y, sigma):
		E = zeros((1,X.shape[0]),float)
		for x_sample_i,x_sample in enumerate(X):
			E[0,x_sample_i] = self._radialfun(x, x_sample, sigma) 
		numerator = dot(E,Y) #numerator.shape (1,2)
		denominator = E.sum() #same as with axis=1 , denominator.shape (1,)
		return np.divide(numerator,denominator) #np.divide(numerator,denominator).shape (1,2)

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

	net = GRNN(alltraindata.indim,alltraindata.outdim)
	Y_predicted = zeros((alltestdata['input'].shape[0],alltestdata['target'].shape[1]))
	sigma = 1.30 # Have to figure out cross-validation to choose sigma!! Though this value gives the best reult!!
	# Every testing data sample is send to .predict along with the training data to get a predicted outcome, a (1,2) vector
	for i,x in enumerate(alltestdata['input']):
		Y_predicted[i] = net.predict(x, alltraindata['input'], alltraindata['target'], sigma)
	Y_predicted = Y_predicted.argmax(axis=1) # Selects the class predicted

	tstresult = percentError(Y_predicted,alltestdata['class'])
	print "Accuracy on test data is: %5.3f%%," % (100-tstresult)

if __name__ == '__main__':
	main()