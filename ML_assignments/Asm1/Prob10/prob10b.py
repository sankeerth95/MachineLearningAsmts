import numpy as np
import data_processing_prob10 as dp
import sys
import matplotlib.pyplot as plt
import utility as utl
#Bayes prior parameters
bet = 0.000
beta_params = [bet, bet]

#0-spam, 1-ham

#docs[word] == no. of files having the word
def add_stat_to_docs(doc, words):
	for word in set(words):
		doc[word] = doc.get(word, 0)+1

def make_docs_hist(X, y, docs):
	j=0
	for x in X:
		if y[j] < 0.5:
			add_stat_to_docs(docs[0], x)
		else:
			add_stat_to_docs(docs[1], x)
		j+=1


def train_bernoulli(X, ylist, V):

	y  = np.array(ylist)
	K = [0.0, 0.0]
	prior = [0.0, 0.0]
	cond_prob = [{}, {}]
	docs = [{}, {}]
	make_docs_hist(X, y, docs)
	for i in xrange(2):

		prior[i] = (1.0*len(y[y==i]))/len(y)
		for word in V:
			cond_prob[i][word] = docs[i].get(word, 0) + beta_params[0]
			K[i] += docs[i].get(word, 0)
		for word in V:
			cond_prob[i][word] /= (K[i]+beta_params[0]+beta_params[1]);

	return prior, cond_prob


def predict_bernoulli(x, prior, cond_prob, V):

	score = [0.0, 0.0]
	for i in xrange(2):
		score[i] += np.log(prior[i])
		for word in V:
			if word in x:
				score[i] += np.log(cond_prob[i].get(word, 1))
			else:
				score[i] += np.log(1-cond_prob[i].get(word, 0))
	if score[0]>score[1]:
		return 0
	else:
		return 1


def predict_bernoulli_test(X, y, prior, cond_prob, V):

	ypredicted = []	
	for x in X:
		ypredicted.append(predict_bernoulli(x, prior, cond_prob, V))
	return ypredicted

in_split = 0
if len(sys.argv) > 1:
	in_split = sys.argv[1]


Xtr, ytr, Xt, yt, vocabulary = dp.get_data(int(in_split))

priorbernoulli, cond_probbernoulli = train_bernoulli(Xtr, ytr, vocabulary)
yprbernoulli = np.array(predict_bernoulli_test(Xt, yt, priorbernoulli, cond_probbernoulli, vocabulary))
print utl.get_perf_params(yprbernoulli, 2*np.array(yt)-1)

betavec = [0.0, 0.0001, 0.50, 10.0, 25, 50.0]
precision = [0.0]*len(betavec)
recall = [0.0]*len(betavec)
f_meas = [0.0]*len(betavec)
accuracy = [0.0]*len(betavec)
j = 0

for beta in betavec:

	Xtr, ytr, Xt, yt, vocabulary = dp.get_data(0)
	beta_params = [beta, beta]
	priorbernoulli, cond_probbernoulli = train_bernoulli(Xtr, ytr, vocabulary)
	yprbernoulli = np.array(predict_bernoulli_test(Xt, yt, priorbernoulli, cond_probbernoulli, vocabulary))
	precision[j], recall[j], f_meas[j], accuracy[j] = utl.get_perf_params(yprbernoulli, 2*np.array(yt)-1)
	print utl.get_perf_params(yprbernoulli, 2*np.array(yt)-1)
	j+=1




plot_data = [(precision[i], recall[i]) for i in xrange(len(betavec))]
plot_data.sort()
print plot_data
x_vals = [plot_data[i][0] for i in xrange(len(betavec))]
y_vals = [plot_data[i][1] for i in xrange(len(betavec))]
plt.figure(1)
plt.plot(x_vals, y_vals)
plt.show()



