import numpy as np

#reads data,labels file into X,y,p
def get_data(features_file, labels_file):

	fdfeat = open(features_file, "r")
	fdlabel = open(labels_file, "r")
	fdfeat.readline()
	fdlabel.readline()

	Xlist = list()
	ylist = list()

	l1features = fdfeat.readline().split(' ')
	l1labels = fdlabel.readline()
	p = int(l1features[1])
	Nsamples = int(l1features[0])

	for i in xrange(Nsamples):
		x=[]
		for j in xrange(p):
			feature_line = fdfeat.readline()
			x.append(float(feature_line))

		Xlist.append(x)
		label_line = fdlabel.readline()
		ylist.append(float(label_line))

	fdlabel.close()
	fdfeat.close()
	
	return (Xlist, ylist, p)

#converts data into DS-2.csv file 
def write_file(out_file, X, y):

	fd = open(out_file, "w")
	j=0
	for x in X:

		line = ""
		for word in x:
			line += str(word)+","
		fd.write(line+str(y[j])+"\n")
		j+=1
	fd.close()


#converts the X,y,p data into libsvm-readable file 
def construct_libsvm_readable(out_file, Xlist, ylist):
	
	fout = open(out_file, "w")
	niter = 0
	for x in Xlist:
		wr_string=""
		if niter > 0:
			wr_string += "\n"+str(y[niter])
		else:
			wr_string += str(y[niter])

		wr_string += " " + str(niter)+":" + str(x)

		fout.write(wr_string)
		niter+=1

	fout.close()


def get_confusion_matrix(ypr, ylist):
	
	confusion = np.zeros((4, 4))
	j = 0
	for y in ypr:
		confusion[y, ylist[j]]+=1
		j+=1

	precision = np.zeros(4)
	recall = np.zeros(4)
	for i in range(4):
		precision[i] = confusion[i][i]/np.sum(confusion[i])
		recall[i] = confusion[i][i]/np.sum(confusion[:, i])
	
	f_measure = 2*precision*recall/(precision+recall)

	return confusion, precision, recall, f_measure



#The X features now have 0 mean and 1 variance over each feature
def standardize(x):
	if np.std(x) > 0:
		z = (x-np.mean(x, axis=0))/np.std(x, axis=0)
		return z
	else:
		return np.ones(len(x))

