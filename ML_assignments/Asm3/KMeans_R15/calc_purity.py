#bash command to be given:
#
#python calc_purity.py <k>
#k: number of clusters in k-means.



import numpy as np
import sys

filename = "R15k" + sys.argv[1]
n_clusters = int(sys.argv[1])

fd = open(filename, "r")

def panic(x):
	print x
	sys.exit()

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
for i in range(15):
	
	line = fd.readline()
	words = line.split(" ")
	
	x=[]
	for word in words[:-2]:
		if word is not '':
			x.append(int(word))
	X.append(x)

fd.close()


X = np.array(X)
print X
print "\n"
sums = 1.0*sum(X)
#print sums

purity = np.zeros(n_clusters)
for i in range(n_clusters):
	x = (np.argmax(X[:, i]))
	purity[i] = X[x, i]/sums[i]

print 100*purity

print np.mean(100*purity)


