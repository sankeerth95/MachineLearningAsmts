import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utility as utl

#get the data andstandardize it.
#store in numpymatrix and array
Xlist, ylist = utl.get_data("./DS3/train.csv", "./DS3/train_labels.csv")
X, y = utl.list_to_standard_form(Xlist, ylist)
Dmatrix = np.transpose(X)*X


#PCR regresses along directions with highest variance, given by direction
#if highest eigenvalue of transpose(X)*X
eigval, eigvec = np.linalg.eig(Dmatrix)
i = eigval.argmax()
vm = eigvec[:, i]
zm = np.array(np.transpose((X*vm)))[0]


#coefficient in this direction is found.
#then the beta vector corresponding to this regression is found.
theta1 = np.sum(zm*y)/np.sum(zm*zm)
yhat = theta1*zm
np.set_printoptions(threshold=np.inf)
beta = theta1*vm


#get the data andstandardize it.
#store in numpymatrix and array
Xlist, ylist = utl.get_data("./DS3/test.csv", "./DS3/test_labels.csv")
X, y = utl.list_to_standard_form(Xlist, ylist)
Dmatrix = np.transpose(X)*X


performance = utl.performance_meas(beta, X, y)
#print performance
utl.write_data("output_prob7.txt", beta, performance)

#norm of a vector
def norm(vec):
	return np.sqrt(vec[0, 0]*vec[0, 0] +vec[1, 0]*vec[1, 0]+vec[2, 0]*vec[2, 0])

vec1 = beta/norm(beta)
vec2 = np.matrix([[1], [0], [0]]) - beta[0, 0]*beta/norm(beta) - beta[1, 0]*beta/norm(beta)
vec2 = vec2/(norm(vec2))


#the test data is projected onto vec1 and vec2 plane
#T is transformation matrix to get points into plane parallel to vec1, vec2
#note that origin is not transformed
#columns of Xproj contains the required projected points in vec1-vec2 basis
T = np.c_[vec2, vec1]
Xproj = X*T

#plot the data. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

#Adjust sparsity in graph
sparse = 2

#get projected points in plane
xs1 = np.array(np.transpose(Xproj[0:-1:sparse, 0]))[0]
ys1 = np.array(np.transpose(Xproj[0:-1:sparse, 1]))[0]
yos = y[0:-1:sparse]

#plot projected points in 2D plot.
#Draw horizontal line as this is the decision boundary.
ax2.plot([-2, 2], [0, 0], c='r')
ax2.scatter(xs1[yos>0], ys1[yos>0] , c='b', marker = 'o')
ax2.scatter(xs1[yos<0], ys1[yos<0], c='r', marker = '^')

#plot scatter plots in 3D
xs = np.array(np.transpose(X[0:-1:sparse, 0]))[0]
ys = np.array(np.transpose(X[0:-1:sparse, 1]))[0]
zs = np.array(np.transpose(X[0:-1:sparse, 2]))[0]

#plot full data in 3D plot. The beta direction is also shown
ax.scatter(xs[yos>0], ys[yos>0], zs[yos>0] , c='b', marker = 'o')
ax.scatter(xs[yos<0], ys[yos<0], zs[yos<0] , c='r', marker = '^')
ax.quiver(0, 0, 0, vm[0], vm[1], vm[2])
plt.show()
