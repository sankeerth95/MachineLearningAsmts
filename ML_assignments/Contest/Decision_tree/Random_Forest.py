#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
#from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import sys

print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

print "Training..."

tree_mod = RandomForestClassifier(min_samples_leaf=16)
tree_mod.fit(Xlist[:27000], ylist[:27000])

print "Training done"

print tree_mod.score(Xlist[27000:], ylist[27000:])
print tree_mod.score(Xlist[:27000], ylist[:27000])


