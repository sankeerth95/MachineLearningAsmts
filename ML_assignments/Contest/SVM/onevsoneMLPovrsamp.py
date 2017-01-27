
import numpy as np
import utility as utl
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import sys


print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

X = np.array(Xlist)
y = np.array(ylist)

print "Training..."

#ovo = OneVsOneClassifier(MLPClassifier(hidden_layer_sizes=(30)), n_jobs=-1)  #48.5 accuracy
ovo = OneVsOneClassifier(MLPClassifier(hidden_layer_sizes=(30), max_iter=400), n_jobs=-1)
ovo.fit(X[:27000], y[:27000])

print "Training done"

print ovo.score(X[27000:], y[27000:])


#print np.mean(cross_val_score(ovo, Xlist, ylist, cv=6))
Xldrbrd = utl.get_data_test("..DATASET/leaderboardTest_data")
y_test=ovo.predict(Xldrboard)


fout=open("sample.txt", "w")

for y1 in y_test:
	fout.write(str(int(y1))+"\n")

fout.close()



