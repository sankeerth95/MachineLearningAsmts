import matplotlib.pyplot as plt
import util as utl
import numpy as np

filepath = "./Clustering_Data/pathbased.txt"

Xlist, ylist, p = utl.get_data_1_file(filepath, separator = '\t')

X = np.array(Xlist)
y = np.array(ylist)


plt.figure()

for class_label in range(6):
	xplot = X[y==class_label][:, 0]
	yplot = X[y==class_label][:, 1]
	plt.plot(xplot, yplot, 'ro')

plt.show()
