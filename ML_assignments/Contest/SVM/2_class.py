#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.qda import QDA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys

i = int(sys.argv[1])
j = int(sys.argv[2])

Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")

X1 = []
y1 = []

hist=[0]*12

k=0
for y in ylist:

	t = int(y)
	hist[t] += 1
	if t == i or t == j:
		X1.append(Xlist[k])		
		y1.append(t)
	k +=1

print hist

X1 = np.array(X1)
y1 = np.array(y1)
#logreg = QDA(reg_param=0.3)
logreg = AdaBoostClassifier()
#logreg = MLPClassifier(hidden_layer_sizes = (30))
#logreg = RandomForestClassifier(min_samples_leaf=10)
logreg.fit(X1[0:4000], y1[0:4000])

print logreg.score(X1[:4000], y1[:4000])
print logreg.score(X1[4000:], y1[4000:])
