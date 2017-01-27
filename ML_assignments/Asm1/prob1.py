#File takes in input
import numpy as np
from scipy import stats as st
import sys


c = 0.3
if len(sys.argv) > 1:
	c = float(sys.argv[1])

feature_size=20
mean1 = np.zeros(feature_size)
mean2 = c*np.ones(feature_size)
diag_elements=np.random.randn(feature_size)

covariance1 = np.diag(diag_elements)
sample_size = 2000
x0f = st.multivariate_normal.rvs(mean1, covariance1, sample_size)
x1f = st.multivariate_normal.rvs(mean2, covariance1, sample_size)
x0 = [[str(x0f[i][j]) for j in xrange(feature_size)] for i in xrange(sample_size)]
x1 = [[str(x1f[i][j]) for j in xrange(feature_size)] for i in xrange(sample_size)]

ftrain = open("DS1-train.csv", "w")
ftest = open("DS1-test.csv", "w")

for i in xrange(sample_size):
	if i < 0.7*sample_size:
		ftrain.write(",".join(x0[i]) + ",-1\n")

	else:
		ftest.write(",".join(x0[i]) + ",-1\n")


for i in xrange(sample_size):
	if i < 0.7*sample_size:
		ftrain.write(",".join(x1[i]) + ",1\n")

	else:
		ftest.write(",".join(x1[i]) + ",1\n")


ftest.close()
ftrain.close()

fd = open("output_prob1.txt", "w")
fd.write("Covariance = diag("+ str(diag_elements)+")")
fd.close()

