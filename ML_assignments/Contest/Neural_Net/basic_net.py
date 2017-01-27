import numpy as np
from scipy.optimize import minimize
import utility as utl
from sklearn.feature_selection import VarianceThreshold

#1/1+exp activation function
def activation(x):
	return 1.0/(1.0+np.exp(-x))

#components along diagonal
def activation_derivative(act_out):
	return act_out*(1.0-act_out)

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

#derivative matrix: jacobian matrix
def softmax_derivative(soft_out):
	t = np.matrix(soft_out)
	return np.diag(soft_out) -np.transpose(t)*t



#Layers are numbered 0, 1, 2... and the 0th node in each layer = 1(constant term)
#
class neural_net:

	def __init__(self, layers, length):
		self.layers = layers		#potential problem: assigning arrays and pointers.
		self.length = length
		self.theta = [np.matrix(np.zeros([length[i+1], 1+length[i]])) for i in xrange(layers-1)]
		#potential problem: interchange i and i+1

	def rand_init_weights(self):
		for i in xrange(self.layers-1):
			self.theta[i] = np.matrix(np.random.rand(length[i+1], 1+length[i]))


def forward_prop(net, x):

	a = [np.zeros(1+net.length[i]) for i in xrange(net.layers-1)]
	a.append(np.zeros(net.length[net.layers-1]))
	out1 = np.ones(len(x)+1)
	out1[1:] = 1.0*x
	a[0] = out1[:]

	for i in xrange(net.layers-2):
		a[i+1][0] = 1.0
#		print np.array(np.matrix(a[i])*np.transpose(net.theta[i]))
		a[i+1][1:] = activation(np.array(np.matrix(a[i])*np.transpose(net.theta[i])))
	
	a[-1][:] = softmax(np.array(np.matrix(a[net.layers-2])*np.transpose(net.theta[net.layers-2])))

	return a
#a[layer][node], a[][0] = 1


def get_loss(net, X, y, gamma=0.0):
	
	RSS = 0.0
	i = 0
	for x in X:
		ypr = forward_prop(net, x)[-1]
		RSS += 0.5*np.sum(np.square(y[i] - ypr))
		i += 1
	
	reg_loss = np.sum(np.sum(np.square(net.theta[0][:, 1:]))) + np.sum(np.sum(np.square(net.theta[1][:, 1:])))

	loss = RSS + gamma*reg_loss
	return loss

#Notation: delta[layer][]
def get_deltas(net, x, y, a):

	delta = [np.zeros(1+net.length[i]) for i in xrange(net.layers-1)]
	delta.append(np.zeros(net.length[net.layers-1]))

	delta[2] = np.array(np.matrix(a[2] - y)*softmax_derivative(a[2]))

	delta[1] = np.array(np.matrix(delta[2])*net.theta[1])*activation_derivative(a[1])
	
	return delta


def backprop_grad(net, Xtrain, ytrain, gamma = 0.0):
	
	D = [0.0]*2
	i = 0
	for x in Xtrain:
		a = forward_prop(net, x)
		delta = get_deltas(net, x, ytrain[i], a)
		D[0] += 1.0*(np.transpose(np.matrix(delta[1][:, 1:])))*np.matrix(a[0])
		D[1] += 1.0*(np.transpose(np.matrix(delta[2])))*np.matrix(a[1])
		i+=1
	
	regularization = net.theta[:]
	regularization[0][:,0] = 0.0
	regularization[1][:,0] = 0.0

	D[0] += 2.0*gamma*regularization[0]
	D[1] += 2.0*gamma*regularization[1]
	
	return D

iterations = 1
current_loss = 0.0
#unroll and solve
def train_net(net, Xtrain, ytrain, gamma=0.0, max_iter=500):

	#Optimization algorithm available in numpy is used here, implimented using gradient methods.
	#BFGS method is used here to minimize loss function
	l = net.length

	theta0 = 0.1*np.random.randn((l[0]+1)*l[1] + (l[1]+1)*l[2])
#	theta0 = np.ones((l[0]+1)*l[1] + (l[1]+1)*l[2])

	def loss_fn(theta):

		global current_loss
		net2 = neural_net(net.layers, net.length)
		col = 0
		net2.theta[0] = 1.0*np.reshape(theta[col:col+(l[0]+1)*l[1]], (l[1], l[0]+1))
		col+=(l[0]+1)*l[1]
		net2.theta[1] = 1.0*np.reshape(theta[col:col+(l[1]+1)*l[2]], (l[2], l[1]+1))
		col+=(l[1]+1)*l[2]
		loss = get_loss(net2, Xtrain, ytrain, gamma)
		current_loss = 1.0*loss
		return loss

	def grad_fn(theta):
		net2 = neural_net(net.layers, net.length)
		col = 0	
		net2.theta[0] = 1.0*np.reshape(theta[col:col+(l[0]+1)*l[1]], (l[1], l[0]+1))
		col+=(l[0]+1)*l[1]
		net2.theta[1] = 1.0*np.reshape(theta[col:col+(l[1]+1)*l[2]], (l[2], l[1]+1))
		col+=(l[1]+1)*l[2]

		D = backprop_grad(net2, Xtrain, ytrain, gamma)
		a1 = np.array(1.0*np.reshape(D[0], (1, l[1]*(l[0]+1))))[0]
		a2 = np.array(1.0*np.reshape(D[1], (1, l[2]*(l[1]+1))))[0]
		jac = np.append(a1, a2)
#		print "grad norm = ", np.linalg.norm(jac)
		return jac


	def check_backprop(eps):
		epsilon = np.zeros(len(theta0))
		der = []
		theta = 1.0*theta0
		for i in xrange(8):
			epsilon[i] = 1.0*eps
			der.append((loss_fn(theta0+epsilon) - loss_fn(theta0-epsilon))/(2*eps))
			epsilon[i] = 0.0

		return np.array(der)


	def gradient_descent(a_des):
		niter = 0
		old_loss = 1.0*loss_fn(theta0)
		theta_old = theta0[:]
		err = 1.0
		while abs(err) > 1e-5:
			theta = theta_old - np.sign(err)*a_des*grad_fn(theta_old)		
			theta_old = theta[:]
			err = loss_fn(theta) - old_loss
			print "ERROR =", err
		return theta

	def fn_cb(xk):

		global iterations
		print "Iteration: ", iterations, " Loss: ", current_loss
		iterations+=1


	grad_calc2 = check_backprop(0.00005)	#real close to actual derivative with as little precision loss
	print "Numerical gradient-backpropagation = ", grad_fn(theta0)[:8]-grad_calc2


	#theta = gradient_descent(1e-3)
	global iterations
	iterations = 1
	out = minimize(loss_fn, theta0,(), method="CG", jac = grad_fn, callback = fn_cb, options={'maxiter':max_iter})
#	print out
	theta = np.transpose(np.matrix(out["x"]))

	col = 0	
	net.theta[0] = 1.0*np.reshape(theta[col:col+(l[0]+1)*l[1]], (l[1], l[0]+1))
	col+=(l[0]+1)*l[1]
	net.theta[1] = 1.0*np.reshape(theta[col:col+(l[1]+1)*l[2]], (l[2], l[1]+1))
	col+=(l[1]+1)*l[2]


def mod_y(ylist):
	y = []
	for y1 in ylist:
		z = np.zeros(4)
		z[y1] = 1
		y.append(z)

	return np.array(y)

def predict(net, x):

	prob = forward_prop(net, x)[-1]
	prob = prob.tolist()
	return prob.index(max(prob))

def get_confusion_matrix(net, X, ylist):
	
	confusion = np.zeros((4, 4))
	j = 0
	for x in X:
		confusion[predict(net, x), ylist[j]]+=1	
		j += 1

	precision = np.zeros(4)
	recall = np.zeros(4)
	for i in range(4):
		precision[i] = confusion[i][i]/np.sum(confusion[i])
		recall[i] = confusion[i][i]/np.sum(confusion[:, i])
	accuracy = sum([confusion[i][i] for i in range(4)])/np.sum(np.sum(confusion))
	f_measure = 2*precision*recall/(precision+recall)

	return confusion, precision, recall, f_measure, accuracy



print "Loading data..."
Xlist, ylist = utl.get_data("../DATASET/train_data", "../DATASET/train_labels")
print "Loaded data"

sel = VarianceThreshold(threshold = 0.8*(1-0.8))
Xreduced = sel.fit_transform(Xlist[:27000])

print len(Xreduced[0])




nx = neural_net(3, [96, 7, 4])
#train_net(nx, X, y, 1e0, max_iter=250)

#prints out confusion matrix, precision, recall, 
#print "Training set parameters:\n", get_confusion_matrix(nx, X, ylist)




#a = forward_prop(n, x)
#print a
#print get_loss(n, [x], [y])

#print get_deltas(n, x, y, a)
#print backprop_grad(n, [x], [y])

#m.probA = prob
#print m
