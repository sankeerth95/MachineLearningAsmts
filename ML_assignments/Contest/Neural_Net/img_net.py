#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import sys

print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

print "Training..."

print "Reducing features..."

#sfm = LDA(n_components=11)
#sfm.fit(Xlist[:27000], ylist[:27000])
#X_reduced = sfm.transform(Xlist[:27000])
#n_features = X_reduced.shape[1]
#print "No.. of features = ", n_features

n_net = MLPClassifier(hidden_layer_sizes = (30))
n_net.fit(Xlist[:27000], ylist[:27000])

print "Training done"

#Xlist1 = sfm.transform(Xlist)
print n_net.score(Xlist[27000:], ylist[27000:])
print n_net.score(Xlist[:27000], ylist[:27000])


