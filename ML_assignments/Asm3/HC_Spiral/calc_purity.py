import numpy as np
import sys
from matplotlib import pyplot as plt


def panic(x):
	print x
	sys.exit()



def get_purity(dist):


	n_classes = 3
	n_instances = 312.0
	filename = "SP_" + dist

	fd = open(filename, "r")
	
	found = False
	while True:

		line = fd.readline()
		if not line: break
	

		if line=="Classes to Clusters:\n":
			found = True
			break

	if not found:
		panic("Parsing error")
	
	fd.readline()
	fd.readline()

	X = []
	for i in range(n_classes):
	
		line = fd.readline()
		words = line.split(" ")
		
		x=[]
		for word in words[:-2]:
			if word is not '':
				x.append(int(word))
		X.append(x)

	accuracy = 0.0
	while True:

		line = fd.readline()
		if not line: break
		word = line.split()
		if not word: continue
		if word[0]=="Incorrectly":
			accuracy = 100*(n_instances-float(word[4]))/n_instances
			break


	fd.close()


	X = np.array(X)
	print X
	print "\n"
	sums = 1.0*sum(X)
	#print sums



	purity = []
	for i in range(len(X[0, :])):
		x = (np.argmax(X[:, i]))
		purity.append(X[x, i]/sums[i])

	purity = np.array(purity)




	return purity, accuracy


purity, accuracy = get_purity(sys.argv[1])
print np.mean(purity)
print accuracy

