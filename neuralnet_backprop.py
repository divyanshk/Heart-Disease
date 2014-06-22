"""Neural Networks BackPRopagation using PyBrain."""

import scipy
import numpy as np
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SigmoidLayer,SoftmaxLayer
from sklearn                     import preprocessing
from sklearn.metrics             import f1_score

in_data=np.genfromtxt('logit-train.csv', delimiter = ',')
out_data = np.genfromtxt('logit-test.csv', delimiter = ',')

in_data=in_data[~np.isnan(in_data).any(1)]
t=len(in_data[0,:])
y_train=np.array(in_data[0:,t-1])
x_train=np.array(in_data[0:,:t-1])

scaler = preprocessing.StandardScaler().fit(x_train) #standardization
#scaler2 = preprocessing.StandardScaler().fit(y_train) #standardization
x_train=scaler.transform(x_train) 
#y_train=scaler2.transform(y_train) 

out_data=out_data[~np.isnan(out_data).any(1)]
t=len(out_data[0,:])
y_test=np.array(out_data[0:,t-1])
x_test=np.array(out_data[0:,:t-1])

x_test=scaler.transform(x_test) 
#y_test=scaler2.transform(y_test) 

print("Building the dataset from CSV files")
#Initialize an empty Pybrain dataset, and populate it
alltraindata=ClassificationDataSet(t-1,1,nb_classes=2)
for count in range(len((in_data))):
    alltraindata.addSample(x_train[count],[y_train[count]])

alltraindata._convertToOneOfMany(bounds=[0,1])

alltestdata=ClassificationDataSet(t-1,1,nb_classes=2)
for count in range(len((out_data))):
    alltestdata.addSample(x_test[count],[y_test[count]])

alltestdata._convertToOneOfMany(bounds=[0,1])
#len(in_data)=152
#alltraindata.indim = 13,alltraindata.outdim = 2
network = buildNetwork( alltraindata.indim, 150 , alltraindata.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer )
trainer = BackpropTrainer( module=network, dataset=alltraindata, momentum=0.1, verbose=False, weightdecay=0.01)
for i in range(20):
  print("Training Epoch #"+str(i))
  trainer.trainEpochs( 1 )

  out = network.activateOnDataset(alltraindata)
  out = out.argmax(axis=1)  # the highest output activation gives the class 
  out = out.reshape(len(in_data),1)

  trnresult = percentError(trainer.testOnClassData(),alltraindata['class'])
  tstresult = percentError(trainer.testOnClassData(dataset=alltestdata),alltestdata['class'])

  print "epoch: %4d" % trainer.totalepochs 
  print "Accuracy on training data is: %5.2f%%," % (100-trnresult), \
  "Accuracy on test data is: %5.2f%%." % (100-tstresult)

average_label = ['micro','macro','weighted']
for label in average_label: 
  f1 = f1_score(y_test, trainer.testOnClassData(dataset=alltestdata), average=label)
  print "f1 score (%s)" %label, "is ", f1
#   trnresult = percentError(trainer.testOnClassData(dataset=alltraindata,verbose=True),y_train)
#	tstresult = percentError(trainer.testOnClassData(dataset=alltestdata,verbose=True),y_test)
#	print "epoch: %4d" % trainer.totalepochs, \
#			"train error: %5.2f%%" % trnresult, \
#			"test error: %5.2f%%" % tstresult
   	
#	ModuleValidator.classificationPerformance(module=network,dataset=alltestdata)
	

