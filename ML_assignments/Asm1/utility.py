import numpy as np

#converts X into matrix, y into standardizearray
def list_to_standard_form(Xlist, ylist):
	X = np.matrix(Xlist)
	X = standardize(X)
	y = np.array(ylist)
	y = standardize(y)
	return X, y

#Gets data from file, parses them and puts it into 
#lists:Xlist and ylist, being feature and output respectively 
def get_data_1_file(in_file):

	fd = open(in_file, "r")
	p = 0
	Xlist = list()
	ylist = list()
	for line in fd:
		words = line.split(',')
		x = [float(w) for w in words[:-1]]
		Xlist.append(x)
		ylist.append(float(words[-1]))
		p = len(x)
	fd.close()
	return Xlist, ylist, p


#get data from feature and label file for classification, into lists
def get_data(file_features, file_labels):
	fdx = open(file_features, "r")
	fdy = open(file_labels, "r")

	Xlist = list()
	ylist = list()
	for line in fdx:
		words = line.split(',')
		x = [float(w) for w in words]
		Xlist.append(x)
		p = len(x)
		ylist.append(float(fdy.readline()[0]))
	#	if ylist[-1] == 2:
	#	break
	fdx.close()
	fdy.close()

	return Xlist, ylist


#write the found coefficients and performance measures onto file
def write_data(out_file, beta, performance):

	fd = open(out_file, "w")

	fd.write("Precision = " + str(performance[0])+"\n")
	fd.write("Recall = " + str(performance[1])+"\n")
	fd.write("F-Score = " + str(performance[2])+"\n")
	fd.write("Accuracy (%correct) = " + str(performance[3])+"\n")
	if len(beta) > 0:
		fd.write("Beta parameters from 0 to p:\n")
		for betacoeff in beta:
			fd.write(str(betacoeff) + "\n")


#Computes performance parameters precision,
#recall and F-measure from learnt model
#TP: true positive, FP: False Positive, TN: true negative, FN: false negative
def performance_meas(beta, Xtest, ytest):
	beta1 = np.transpose(beta)
	ypredicted = np.array(beta1*np.transpose(Xtest))[0] > 0.0
	return get_perf_params(ypredicted, ytest)
#	print ypredicted
	TP = 0.0
	FP = 0.0
	FN = 0.0
	TN = 0.0
	j=0
	for y in ytest:
		if y>0:
			if ypredicted[j]>0:
				TP+=1
			else:
				FN+=1
		else:
			if ypredicted[j]>0:
				FP+=1
			else:
				TN+=1
		j += 1

	precision = 1.0
	if (TP+FP) != 0:
		precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	F_score = 2*precision*recall/(precision+recall)
	accuracy = 100*(TP+TN)/(TP+TN+FP+FN)
	return precision, recall, F_score, accuracy

#get parameters for 2-class classification problems
def get_perf_params(ypredicted, ytest):
	TP = 0.0
	FP = 0.0
	FN = 0.0
	TN = 0.0
	j=0
	for y in ytest:
		if y>0:
			if ypredicted[j]>0:
				TP+=1
			else:
				FN+=1
		else:
			if ypredicted[j]>0:
				FP+=1
			else:
				TN+=1
		j += 1

	precision = 1.0
	if (TP+FP)!=0:
		precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	F_score = 2*precision*recall/(precision+recall)

	ypred = 1*ypredicted
	for i in xrange(len(ypredicted)):
		if ypredicted[i] == 0:
			ypred = 2*ypredicted-1

	accuracy = 100*(TP+TN)/(TP+TN+FP+FN)
	return precision, recall, F_score, accuracy
#The X features now have 0 mean and 1 variance over each feature
def standardize(x):
	if np.std(x) > 0:
		z = (x-np.mean(x, axis=0))/np.std(x, axis=0)
		return z
	else:
		return np.ones(len(x))
