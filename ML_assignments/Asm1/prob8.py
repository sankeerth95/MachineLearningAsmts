import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utility as utl


#Extract and standardize data from file for training.
Xlist, ylist = utl.get_data("./DS3/train.csv", "./DS3/train_labels.csv")
X, y = utl.list_to_standard_form(Xlist, ylist)
Dmatrix = np.transpose(X)*X


N = len(X)

#Store means of distribution
u_1 = np.mean(X[y>0], axis=0)
u_2 = np.mean(X[y<0], axis=0)

#Build between class variance matrix
Sb = np.transpose(u_1-u_2)*(u_1-u_2)

#Build within class variance matrix
M1 = X[y>0]-u_1
M2 = X[y<0]-u_2
Sw = np.transpose(M1)*M1 + np.transpose(M2)*M2


#on maximizing between class variance by within class variance
# the direction corresponds to max.eigenvalue of inv(Sw)*Sb 
eig_val, eig_vec = np.linalg.eig(np.linalg.inv(Sw)*Sb)
w = eig_vec[:, -1]

#Find u corresponding to this direction
zm = np.array(np.transpose((X*w)))[0]

#find the coefficients in it's expansion, and print out the values
theta1 = np.sum(zm*y)/np.sum(zm*zm)
yhat = theta1*zm
np.set_printoptions(threshold=np.inf)
beta = theta1*w
#print "beta = ", beta

#norm of a vector
def norm(vec):
	return np.sqrt(vec[0, 0]*vec[0, 0] +vec[1, 0]*vec[1, 0]+vec[2, 0]*vec[2, 0])


#vec1 and vec2 are unit vectors in the plane containing beta.
#here, one is along beta, other is perpendicular to it
#Hence classifying boundary is along vec2 in this plane.
vec1 = beta/norm(beta)
vec2 = np.matrix([[1], [0], [0]]) - beta[0, 0]*beta/norm(beta) - beta[1, 0]*beta/norm(beta)
vec2 = vec2/(norm(vec2))

#get test-set data
Xlist, ylist = utl.get_data("./DS3/train.csv", "./DS3/train_labels.csv")
X, y = utl.list_to_standard_form(Xlist, ylist)
Dmatrix = np.transpose(X)*X

#get performance measure and print them
performance = utl.performance_meas(beta, X, y)
#print performance

utl.write_data("output_prob8.txt", beta, performance)

#the test data is projected onto vec1 and vec2 plane
#T is transformation matrix to get points into plane parallel to vec1, vec2
#note that origin is not transformed
#columns of Xproj contains the required projected points in vec1-vec2 basis
T = np.c_[vec2, vec1]
Xproj = X*T

#plot the data. 
fig = plt.figure()
fig2 = plt.figure()

ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111, projection='3d')

#Adjust sparsity in graph
sparse = 2

#get projected points in plane
xs1 = np.array(np.transpose(Xproj[0:-1:sparse, 0]))[0]
ys1 = np.array(np.transpose(Xproj[0:-1:sparse, 1]))[0]

#plot scatter plots in 3D
xs = np.array(np.transpose(X[0:-1:sparse, 0]))[0]
ys = np.array(np.transpose(X[0:-1:sparse, 1]))[0]
zs = np.array(np.transpose(X[0:-1:sparse, 2]))[0]

#plot projected points in 2D plot
#Draw horizontal line as this is the decision boundary.
yos = y[0:-1:sparse]
ax.plot([-2, 2], [0, 0], c='r')
ax.scatter(xs1[yos>0], ys1[yos>0] , c='b', marker = 'o')
ax.scatter(xs1[yos<0], ys1[yos<0], c='r', marker = '^')


#plot full data in 3D plot. The beta direction is also shown
ax2.scatter(xs[yos>0], ys[yos>0], zs[yos>0] , c='b', marker = 'o')
ax2.scatter(xs[yos<0], ys[yos<0], zs[yos<0] , c='r', marker = '^')
ax2.quiver(0, 0, 0, beta[0], beta[1], beta[2])
#ax2.quiver(0, 0, 0, w[0], w[1], w[2])
plt.show()
