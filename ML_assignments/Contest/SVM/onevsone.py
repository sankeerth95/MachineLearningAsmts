#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import sys

print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

print "Training..."


ovo = OneVsOneClassifier(SVC(C=1.0), n_jobs=-1)
ovo.fit(Xlist[:25000], ylist[:25000])

print "Training done"

print ovo.score(Xlist[25000:], ylist[25000:])




