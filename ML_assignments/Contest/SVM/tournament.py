#Binary class classifier: Outputs thetraining error
#Also outputs histogram of class

import numpy as np
import utility as utl
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import sys

print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

def train_logreg(i, j, Xlist, ylist):

	X1 = []
	y1 = []
	k=0
	for y in ylist:
	
		X1.append(Xlist[k])
		t = int(y)
		y1.append(t)
		k+=1

	X1 = np.array(X1)
	y1 = np.array(y1)
	logreg = RandomForestClassifier(min_samples_leaf=5)
	logreg.fit(X1, y1)
	return logreg


print "Training..."
train_matrix = [[None]*12]*12
it=0
for i in xrange(12):
	for j in xrange(12):
		if i < j:
			print it
			train_matrix[i][j] = train_matrix[j][i] = train_logreg(i, j, Xlist[0:25000], ylist[0:25000])
			it+=1
print "Training done"


x = np.array(Xlist[-1])
#preliminary
pre_res = [0]*6
pre_res[0] = train_matrix[0][1].predict(x)
pre_res[1] = train_matrix[2][3].predict(x)
pre_res[2] = train_matrix[4][5].predict(x)
pre_res[3] = train_matrix[6][7].predict(x)
pre_res[4] = train_matrix[8][9].predict(x)
pre_res[5] = train_matrix[10][11].predict(x)

print pre_res

#quarter
quart_res = [0]*3
quart_res[0] = train_matrix[pre_res[0]][pre_res[1]].predict(x)
quart_res[1] = train_matrix[pre_res[2]][pre_res[3]].predict(x)
quart_res[2] = train_matrix[pre_res[4]][pre_res[5]].predict(x)

print quart_res

#semi
semi_res = [0]*2
semi_res[0] = quart_res[0]
semi_res[1] = train_matrix[quart_res[1]][quart_res[2]].predict(x)

print semi_res

#final
final_res = train_matrix[semi_res[0]][semi_res[1]].predict(x)
print ylist[-1], final_res


