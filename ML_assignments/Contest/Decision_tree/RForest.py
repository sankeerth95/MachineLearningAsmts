#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys

print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

print "Training..."

print "Reducing features..."

sfm = SelectFromModel(SVC(C=1.0), threshold=0.0)
sfm.fit(Xlist[:2700], ylist[:2700])
X_reduced = sfm.transform(Xlist[:27000])
n_features = X_reduced.shape[1]
print "No.. of features = ", n_features

tree_mod = RandomForestClassifier(min_samples_leaf=5)

tree_mod.fit(X_reduced, ylist[:27000])

print "Training done"

Xlist1 = sfm.transform(Xlist)
print tree_mod.score(Xlist1[27000:], ylist[27000:])
print tree_mod.score(Xlist1[:27000], ylist[:27000])


