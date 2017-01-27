import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from svmutil import *
from svm import *
import util as utl

#set parameters for SVM:
def get_model((Xlist, ylist), kern, C=1e0, gamma=5e-3, degree=4):

	prob  = svm_problem(ylist, Xlist, isKernel=True)

	param1 = svm_parameter()
	param1.kernel_type = kern
	param1.cross_validation=False
	param1.degree = degree
	param1.C = C
	param1.gamma=gamma

	m = svm_train(prob, param1)
	return m

#performing 5-foldcross validation over the generated shuffle dataset, taking 200 datapoints at a time.
#test this data with the remaining samples
#The best model m is selected after the 5-fold cross validation
def get_model_par(kern, X, y, C=1e0, gamma=5e-3, degree=4):
	m1 = []
	ACC_max = 0
	for i in [0, 200, 400, 600, 800]:
		m1.append(get_model((X.tolist()[i:i+200], y[i:i+200]), kern, C, gamma, degree))
		print i
		y1 = y[i+200:]
		for y11 in y[0:i]:
			y1.append(y11)
	
		X1 =X.tolist()[i+200:]
		for x11 in X.tolist()[0:i]:
			X1.append(x11)

		pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(y1, X1, m1[-1])
	#confusion, precision, recall, f_score = utl.get_confusion_matrix(pred_labels, y[i:i+200])
		if ACC > ACC_max:
			m = m1[-1]
			ACC_max = ACC*1.0
	return m



#obtain training and test datainto Xrlist and yrlisr
Xrlist, yrlist, pr = utl.get_data("Train_features", "Train_labels")
Xr = utl.standardize(np.matrix(Xrlist))

#get the test data into Xtlist and ytlist
Xtlist, ytlist, pt = utl.get_data("Test_features", "Test_labels")
Xt = utl.standardize(np.matrix(Xtlist))

#X and y are the shuffled dataset.

X, y = shuffle(Xr, yrlist)
utl.write_file("Shuffle_train.csv", X.tolist(), y)


ml = get_model_par(LINEAR, X, y, 1e0)
mp = get_model_par(POLY, X, y, C=1e3, degree=4)
mg = get_model_par(RBF, X, y, C=1e0, gamma=5e-3)
ms = get_model_par(SIGMOID, X, y ,C=1e0)


#save the matlab file
sio.savemat("EE13B102.mat", {"model1":ml, "model2":mp, "model3":mg, "model4":ms})

#evaluate performance parameters for the test set
pred_labelst, (ACCt, MSEt, SCCt), pred_valuest = svm_predict(ytlist, Xt.tolist(), ml)
confusiont, precisiont, recallt, f_scoret = utl.get_confusion_matrix(pred_labelst, ytlist)

print confusiont
print "Precision: \n", precisiont
print "Recall: \n", recallt
print "f_score: \n", f_scoret


