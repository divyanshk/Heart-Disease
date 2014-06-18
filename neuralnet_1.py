"""Neural Networks using PyBrain."""

from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import ModuleValidator
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np

in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

in_data=in_data[~np.isnan(in_data).any(1)]
t=len(in_data[0,:])
y_train=np.array(in_data[0:,t-1])
x_train=np.array(in_data[0:,:t-1])

out_data=out_data[~np.isnan(out_data).any(1)]
t=len(out_data[0,:])
y_test=np.array(out_data[0:,t-1])
x_test=np.array(out_data[0:,:t-1])

y_train = y_train.reshape( -1, 1 )  

ds1 = SupervisedDataSet( t-1, 1 )
ds1.setField( 'input', x_train )
ds1.setField( 'target', y_train )

y_test = y_test.reshape( -1, 1 )  

ds2 = SupervisedDataSet( t-1, 1 )
ds2.setField( 'input', x_test )
ds2.setField( 'target', y_test )

hidden_layer_size=200

net = buildNetwork( t-1 , hidden_layer_size, 1 ,bias = True )
trainer = BackpropTrainer( module=net, dataset=ds1, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(20):
	print("Training Epoch #"+str(i))
   	trainer.trainEpochs( 1 )

	p = net.activateOnDataset(ds2)
	p = p.argmax(axis=1)  # the highest output activation gives the class
	p = p.reshape(-1,-1) 
	print p

hitrate = ModuleValidator.classificationPerformance(module=net,dataset=ds1)
print hitrate

#trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 20)

# for i in range(20):
# 	trainer.trainEpochs(3)
# 	trnresult = percentError(trainer.testOnClassData(),y_train)
# 	tstresult = percentError(trainer.testOnClassData(dataset=ds2),y_test)
# 	print "epoch: %4d" % trainer.totalepochs, \
# 			"train error: %5.2f%%" % trnresult, \
# 			"test error: %5.2f%%" % tstresult
# 	out = net.activateOnDataset(ds1)
# 	out = out.argmax(axis=1)  # the highest output activation gives the class
# 	#print out
# 	figure(1)
# 	ioff()  # interactive graphics off
# 	clf()   # clear the plot
# 	hold(True) # overplot on
# 	for c in [0, 1]:
# 		here, _ = where(y_test == c)
# 		plot(x_test[here, 0], x_test[here, 1], 'o')
# 	if out.max() != out.min():  # safety check against flat field
# 		contourf(X, Y, out)   # plot the contour
# 	ion()   # interactive graphics on
# 	draw()  # update the plot


# """ Finally, keep showing the plot until user kills it. """
# ioff()
# show()

print "Done!!"