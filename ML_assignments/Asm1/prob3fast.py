import numpy as np
import utility as utl
from sklearn.neighbors import KNeighborsClassifier
import sys
from matplotlib import pyplot as plt
#Faster program using library from sklearn
k = 10
if len(sys.argv) > 1:
	k = int(sys.argv[1])

#get test set prediction for a set of sample points in Xtest and compare
#with the actual outputs in ytest. Them get performance measure
#train_data is a tuple of training data (Xtrain, ytrain)
def performance_meas(Xtest, ytest, nbrs):

	testset_len = len(Xtest)
#	testset_len = 100
	ypredicted = np.zeros(testset_len)
	for i in xrange(testset_len):
		x = np.array(Xtest[i, :])[0]
		ypredicted[i] = 0.5*(nbrs.predict(x)+1)

	return utl.get_perf_params(ypredicted, ytest)

def try_out(Xtrain, ytrain, Xt, yt, klist):

	precis = [0]*len(klist)
	recall =[0]*len(klist) 
	f_meas = [0]*len(klist) 
	acc = [0]*len(klist)
	j = 0
	for i in klist:
		#fit data into Tree
		nbrs = KNeighborsClassifier(n_neighbors=i+1)
		nbrs.fit(Xtrain, ytrain)
		#Get performance measure and write them to file "output_prob3.txt"
		(precis[j], recall[j], f_meas[j], acc[j]) = performance_meas(Xt, yt, nbrs)
		del nbrs
		j += 1
	plt.figure(1)
	plt.plot(klist, precis)
	plt.title("Precision")
	plt.xlabel("k")

	plt.figure(2)
	plt.plot(klist, recall)
	plt.title("Recall")
	plt.xlabel("k")

	plt.figure(3)
	plt.plot(klist, f_meas)
	plt.title("F Measure")
	plt.xlabel("k")
	
	plt.figure(4)
	plt.plot(klist, acc)
	plt.title("Accuracy")
	plt.xlabel("k")
	plt.show()
		
		
	

#The lists are converted into numpy matrices and arrays
Xlist, ylist, p = utl.get_data_1_file("DS1-train.csv")
Xtrain = np.matrix(Xlist)
ytrain = np.array(ylist)


#fit data into Tree
nbrs = KNeighborsClassifier(n_neighbors=k)
nbrs.fit(Xtrain, ytrain)

#get the test data in matrix and array form
Xlistt, ylistt, pt = utl.get_data_1_file("DS1-test.csv")
Xt = np.matrix(Xlistt)
yt = np.array(ylistt)

#Get performance measure and write them to file "output_prob3.txt"
performance = performance_meas(Xt, yt, nbrs)
utl.write_data("output_prob3.txt", [],  performance)

klist = np.arange(1, 60, 2)
try_out(Xtrain, ytrain, Xt, yt, klist)




