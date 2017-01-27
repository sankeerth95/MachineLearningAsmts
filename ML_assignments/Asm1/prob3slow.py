import numpy as np
import utility as utl
#Running the program takes atleast 1 minute, since for this large data set,
#iach test data is iterated over every point to find nearest neighbors.
# A wiser approach is to use a tree data structure for Nearest Neighbors
#to reduce time

#distance measure implimented as the squared distance between the vectors
def dist(x1, x2):
	x = x1-x2
	return np.sum(x*x)

#get test set prediction for a set of sample points in Xtest and compare
#with the actual outputs in ytest. Them get performance measure
#train_data is a tuple of training data (Xtrain, ytrain)
def performance_meas(k, Xtest, ytest, train_data):

	testset_len = len(Xtest)
#	testset_len = 100
	ypredicted = np.zeros(testset_len)
	for i in xrange(testset_len):
		x = np.array(Xtest[i, :])[0]
		ypredicted[i] = kNN_predict(k, x, train_data)

	return utl.get_perf_params(ypredicted, ytest)


#Simple iteration over all training data points performed and the
#k-nearest neighbor majority is given as the output
#train_data is a tuple of (Xtrain, ytrain). x is the point whose
#y is to be predicted. k is no. of neighbors to consider
def kNN_predict(k, x, train_data):

	Xtrain, ytrain = train_data
	distance_list = []
	for i in xrange(len(ytrain)):
		xtemp = np.array(Xtrain[i, :])[0]
		distance_list.append((dist(x, xtemp), y[i]))
#	print distance_list
	distance_list.sort()
	k_list = distance_list[:k+1]
	score = sum([k_list[i][1] for i in xrange(k)])
	if score > 0.0:
		return 1
	else:
		return -1

#The lists are converted into numpy matrices and arrays
Xlist, ylist, p = utl.get_data_1_file("DS1-train.csv")
X = np.matrix(Xlist)
y = np.array(ylist)


#get the test data in matrix and array form
Xlistt, ylistt, pt = utl.get_data_1_file("DS1-test.csv")
Xt = np.matrix(Xlistt)
yt = np.array(ylistt)

#Get performance measure and write them to file "output_prob3.txt"
performance = performance_meas(10, Xt, yt, (X, y))
utl.write_data("output_prob3.txt", [],  performance)


