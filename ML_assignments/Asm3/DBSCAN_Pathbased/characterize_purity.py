#bash command to be given:
#
#python calc_purity.py <k>
#k: number of clusters in k-means.



import numpy as np
import sys
from matplotlib import pyplot as plt

m = int(sys.argv[1])

def panic(x):
	print x
	sys.exit()



def get_purity(m, e):

	filename = "PBm" + str(m) + "e" + str(e)

	fd = open(filename, "r")
	
	unclustered = 0.0
	found = False
	while True:

		line = fd.readline()
		if not line: break
	
		word = line.split()
		if not word: continue
		if word[0] == "Unclustered":
			unclustered = float(word[3])

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

	accuracy = 0.0
	while True:

		line = fd.readline()
		if not line: break
		word = line.split()
		if not word: continue
		if word[0]=="Incorrectly":
			accuracy = 100*(372-float(word[4])-unclustered)/372.0
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


mean_purity = []
accuracy = []



if m==3:
	e = [ 0.03, 0.04, 0.05,0.06, 0.07, 0.08] #m=3

elif m==6:
	e = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09] #m=6 

elif m==9:
	e = [0.04, 0.05, 0.06, 0.07, 0.08]
else: panic("DBSCAN data not present")

e.sort()

for i in e:
	purity, acc = get_purity(m, i)
	mean_purity.append(100*np.mean(purity))
	accuracy.append(acc)


plt.figure(1)
plt.plot(e, mean_purity)
plt.xlabel('epsilon')
plt.ylabel('Average Purity')
plt.ylim(ymax = 110, ymin = -10)

plt.figure(2)
plt.plot(e, accuracy)
plt.xlabel('epsilon')
plt.ylabel('Accuracy')
plt.ylim(ymax = 110, ymin = -10)
plt.show()






