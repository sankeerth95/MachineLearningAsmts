#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import sys

i = int(sys.argv[1])
j = int(sys.argv[2])

Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")


trainperf = np.zeros((12, 12))
testperf = np.zeros((12, 12))
iteration = 0
for i in range(12):
	for j in range(i):
	
		iteration+=1
		print iteration
		X1 = []
		y1 = []

		k=0
		for y in ylist:
	
			t = int(y)
			if t==i or t==j:
				X1.append(Xlist[k])		
				y1.append(t)
			k +=1


		X1 = np.array(X1)
		y1 = np.array(y1)


		clf = AdaBoostClassifier(RandomForestClassifier(min_samples_leaf=20))
#		clf = MLPClassifier(hidden_layer_sizes = (40), alpha=0.1)
#		clf = SVC(C=1.0) #0.82 accuracy
#clf = RandomForestClassifier(min_samples_leaf=5, n_estimators = 100) #16 gives best
	#	clf = QDA(reg_param=0.4)
#clf = LDA(shrinkage='auto', solver='eigen')
		clf.fit(X1[0:4000], y1[0:4000])
#print clf.predict(X1[4:9]), y1[4:9]
		trainperf[i, j] = clf.score(X1[:4000], y1[:4000])
		testperf[i, j]  = clf.score(X1[4000:], y1[4000:])

#print np.mean(cross_val_score(clf, X1, y1, cv=5))


print "hidden nodes = 30, alpha = 1"
print trainperf
print testperf


