#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys

i = int(sys.argv[1])
j = int(sys.argv[2])

Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")

X1 = []
y1 = []
X2 = []
y2 = []

hist=[0]*12

k=0
for y in ylist:

	t = int(y)
	hist[t] += 1
	if t==i:
		X1.append(Xlist[k])		
		y1.append(t)

	if t==j:
		X2.append(Xlist[k])		
		y2.append(t)

	k +=1

print hist

X1 = np.array(X1)
y1 = np.array(y1)
X2 = np.array(X2)
y2 = np.array(y2)

#clf = SVC(C=1.0) #0.82 accuracy
#clf = RandomForestClassifier(min_samples_leaf=5, n_estimators = 100) #16 gives best
clf1 = GaussianMixture()
clf2 = GaussianMixture()
#clf = LDA(shrinkage="auto", solver)
clf1.fit(X1[0:2000])
clf2.fit(X2[0:2000])
#print clf.predict(X1[4:9]), y1[4:9]
print clf1.score(X1[2000:], y1[2000:])
print clf2.score(X2[2000:], y2[2000:])


#print np.mean(cross_val_score(clf, X1, y1, cv=5))

