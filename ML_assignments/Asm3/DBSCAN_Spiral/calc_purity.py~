#bash command to be given:
#
#python calc_purity.py <m> <e>
#m: min-points, e:epsilon.



import numpy as np
import sys

filename = "PB" + sys.argv[1]+"e"+sys.argv[2]
min_points = int(sys.argv[1])

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
for i in range(2):
	
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

sums = 1.0*sum(X)
print sums

purity = []
for i in range(len(X[0, :])):
	x = (np.argmax(X[:, i]))
	purity.append(X[x, i]/sums[i])

purity = np.array(purity)
print 100*purity


