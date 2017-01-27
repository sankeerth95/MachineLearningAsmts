import numpy as np
from scipy.optimize import minimize
import sys
import utility as utl
#from sklearn import linear_model

# The linear regression coefficients have been found using three approaches, one with optimizing
#squared error loss with gradient methods, one by using the analytical solution, and the other using python library sklearn

lambda1 = 0.0
if len(sys.argv) > 1:
	lambda1 = sys.argv[1]

#beta, x is numpy array and matrix respectively, so is y
def lsloss(beta, X, y):
	pred_error = np.array(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)
	sum_of_squares = np.sum(pred_error*pred_error)
	regularization = lambda1*np.sum(beta[1:]*beta[1:])
	return sum_of_squares+regularization

#create the gradient ,ethod to use gradient descent
def lsgrad(beta, X, y):
	gradient = np.zeros(len(beta))
	gradient[1:] = 2.0*(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)*X + lambda1*beta[1:]
	gradient[0] = 2.0*np.sum(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)	
	return gradient


#Obtains data and converys the features into numpy matrix
#the y
Xlist, ylist, p = utl.get_data_1_file("DS1-train.csv")
X = np.matrix(Xlist)
y = np.array(ylist)

beta0 = np.zeros(p+1)
#Optimization algorithm available in numpy is used here, implimented using gradient methods.
#BFGS method is used here to minimize loss function
out = minimize(lsloss, beta0, (X, y), method='BFGS', jac = lsgrad)
beta = np.transpose(np.matrix(out["x"]))

#Set test data in required form, i.e. numpy arrays, matrices
Xlistt, ylistt, pt = utl.get_data_1_file("DS1-test.csv")
Xt = np.matrix(Xlistt)
yt = np.array(ylistt)

#get performance measures
Xt1 = np.c_[np.ones(len(yt)), Xt]
performance = utl.performance_meas(beta, Xt1, yt)
utl.write_data("output_prob2.txt", beta, performance)
