import numpy as np
import random as rnd
from scipy.optimize import minimize
#from sklearn import linear_model


lambda1 = 0.0
N = 1
#beta, x is numpy array and matrix respectively, so is y
def lsloss(beta, X, y):
	pred_error = np.array(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)
	sum_of_squares = (1.0/N)*np.sum(pred_error*pred_error)
	regularization = 0.5*lambda1*np.sum(beta[1:]*beta[1:])
	return sum_of_squares+regularization

#lsgrad calculates gradient of the RSS terms, require for optimization
def lsgrad(beta, X, y):
	gradient = np.zeros(len(beta))
	gradient[1:] = (2.0/N)*(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)*X + lambda1*beta[1:]
	gradient[0] = (2.0/N)*np.sum(np.matrix(beta[1:])*np.transpose(X) + beta[0] - y)	
	return gradient

#function shuffles data and selects first 80% as train,
#the rest as test cases. the file is stored with filename out_file_train
def make_splits(in_filename, out_file_train, out_file_test):

	fd = open(in_filename, "r")
	lines = [line for line in fd]
	rnd.shuffle(lines)
	a = int(0.8*len(lines))
	
	fdwtr = open(out_file_train, "w")
	fdwts = open(out_file_test, "w")
	i = 0
	for line in lines:
		if i < a:
			fdwtr.write(line)
		else:
			fdwts.write(line)
		i+=1
	fdwtr.close()
	fdwts.close()


#gets data from file into lists Xlist and ylist.
def get_data(filename):
	fd = open(filename, "r")
	beta0 = 0
	p = 0
	Xlist = list()
	ylist = list()
	for line in fd:
		words = line.split(',')
		x = [float(w) for w in words[5:-1]]
		Xlist.append(x)
		ylist.append(float(words[-1]))
		p = len(x)
	fd.close()
	return Xlist, ylist, p

#writes the coefficients learned, and the RSS error
def write_data(out_file, beta, RSS):

	fd = open(out_file, "w")

	for i in xrange(5):
		fd.write("RSS error " + str(i+1)+"= " + str(RSS[i])+"\n")

	fd.write("RSS error average = " + str(0.2*sum(RSSerror)) + "\n")

	for beta1 in beta:
		fd.write("Beta parameters from 0 to p:\n")
		for betacoeff in beta1:
			fd.write(str(betacoeff) + "\n")
		fd.write("\n")
	fd.close()



#PROGRAM STARTS HERE
#loop over 5 times to create 5 randomly generated 80-20 splits.
#Solve for beta by optimizing in each case, and  gives the RSS error
# for the 20% of remaining data (test data)
RSSerror = []
betalist = []
for i in xrange(5):
	
	make_splits("communities_usable.data", "CandC-train"+str(i+1)+".csv", "CandC-test"+str(i+1)+".csv")

	Xlist, ylist, p = get_data("CandC-train" + str(i+1) + ".csv")
	beta0 = np.zeros(p+1)
	if i == 0:
		beta = np.zeros(p+1)
	X = np.matrix(Xlist)
	y = np.array(ylist)

	out = minimize(lsloss, beta0, (X, y), method='BFGS', jac = lsgrad)
	beta = out['x']
	betalist.append(beta.tolist())

	Xlistt, ylistt, pt = get_data("CandC-test"+str(i+1)+".csv")
	Xt = np.matrix(Xlistt)
	yt = np.array(ylistt)

	RSSerror.append(lsloss(beta, Xt, yt))


print RSSerror
print "Avg = ", 0.2*sum(RSSerror)

write_data("output_prob5.txt", betalist, RSSerror)




